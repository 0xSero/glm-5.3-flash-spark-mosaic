"""Independent diagnostic oracle for BF16/SILU NVFP4 W4A16 packed routes.

Scope: raw, pre-repacking NVFP4 arrays; gate-then-up W13; no rotation,
no input route weighting, no fused atomic top-k sum, no distributed reduction.
This is NOT a new acceptance tolerance or a replacement quantization metric.
Keep the all-FP32 dequantized-weight oracle as a separate diagnostic.

Source: B12x 3b862805d2b7fd52e6fe507fd28038bd48797cf5:
  prepare.py:363-413,758-797: power-of-two C, block cutoff, packed global.
  intrinsics.py:3061-3097,3139-3176: register E2M1/E4M3 decoding.
  kernel.py:1858-1945,5087-5100,5227-5270: FC1/FC2 scaling and BF16 stores;
             7747-7781: SILU rounding; 8411-8432: ordered FP32 top-k sum.

For each projection C is shared across all experts in its scale tensor.
Let S'=S when S*C >= 1/64, otherwise 0. Packed register weights represent
BF16(Q*S'*C*2^-119); the corresponding FP32 global is G*2^119/C.
Away from underflow/overflow their powers of two cancel, giving normalized
BF16(Q*S') weights with global G. However, FC2 rounds the INTERNAL dot product
to BF16 before multiplying BF16(packed_global * route_weight). This helper
keeps the literal packed scale domain to preserve those rounding boundaries.
It does not call any B12x kernel, packing or dequantization helper.

FP32 BLAS accumulation order and torch.exp need not bitwise reproduce CUDA
MMA/fast-math exp. These residual differences still need the original unchanged
acceptance checks. This reference is eager-only, not CUDA-graph-safe.

CPU checks: python reference_packed.py --self-test
"""
import math
import unittest

import torch
import torch.nn.functional as F

B12X_COMMIT = "3b862805d2b7fd52e6fe507fd28038bd48797cf5"
SOURCE_SHA256 = {
    "b12x/moe/_shared/kernels/w4a16/prepare.py":
        "554f3d9a8e8843fde1e2b086c667fa8778bd51acde635c7b831ea9157ac2ecce",
    "b12x/moe/_shared/kernels/w4a16/kernel.py":
        "c2c18fc5964a832256179a6d0dba127213592de69c1fd907d3855fe97c3bd821",
    "b12x/_lib/intrinsics.py":
        "7e0c8e42bcdf609f2aae29a27c9f9013e2a8d3da91f8e032af234554d206ef7a",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def scale_factor(scales):
    """Projection-wide C; never compute it independently for each expert."""
    _require(scales.device.type == "cpu", "raw scales must be on CPU")
    values = scales.float()
    _require(bool(torch.isfinite(values).all()) and bool((values >= 0).all()),
             "scales must be finite and nonnegative")
    maximum = float(values.max())
    _require(maximum <= 448, "expected original E4M3 block scales")
    return float(2 ** math.floor(math.log2(448 / maximum))) if 0 < maximum < 448 else 1.


def decode_register_weight(packed, scales, factor):
    """CPU nibble/LUT decode to actual BF16 register-weight scale domain."""
    _require(packed.device.type == scales.device.type == "cpu", "raw arrays must be CPU")
    _require(packed.dtype == torch.uint8 and packed.ndim == 2, "expected uint8 [N,K/2]")
    n, half_k = packed.shape
    _require(half_k % 8 == 0 and tuple(scales.shape) == (n, half_k // 8),
             "expected group-16 block scales")
    _require(factor >= 1 and math.log2(factor).is_integer(), "C must be a power of two")
    lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                        -0., -.5, -1., -1.5, -2., -3., -4., -6.], dtype=torch.float32)
    codes = torch.stack((packed & 15, packed >> 4), -1).reshape(n, -1, 16)
    s = scales.float() * factor
    s = torch.where(s >= 1 / 64, s, 0.)
    # E2M1 exponent bias is -126, packed scale contributes an extra 2^7.
    return (lut[codes.long()] * s.unsqueeze(-1) * (2. ** -119)).reshape(n, -1).bfloat16()


def silu_bf16(gate, up, limit=10.):
    """FC1 inputs and SILU term are separately rounded before multiplication."""
    g, u = gate.bfloat16().float(), up.bfloat16().float()
    hits = {"gate_above_10": int((g > limit).sum()),
            "up_outside_10": int((u.abs() > limit).sum())}
    g, u = g.clamp(max=limit), u.clamp(min=-limit, max=limit)
    silu = (g * (1. / (1. + torch.exp(-g)))).bfloat16()
    return (silu.float() * u.bfloat16().float()).bfloat16(), hits


def ordered_sum(route_outputs):
    """[M,top_k,H] BF16 partials, accumulated in original route-slot order."""
    _require(route_outputs.dtype == torch.bfloat16, "route outputs must be BF16")
    result = torch.zeros_like(route_outputs[:, 0], dtype=torch.float32)
    for slot in range(route_outputs.shape[1]):
        result += route_outputs[:, slot].float()
    return result.bfloat16()


@torch.inference_mode()
def reference_packed(x, ids, weights, raw, *, shared=None, swiglu_limit=10.):
    """Return (routed_BF16_or_shared_sum_BF16, diagnostics).

    x/ids/weights use the same device; raw is the six pre-setup CPU tensors
    already collected by gpu_component.py. ids address the raw expert axis.
    Optional shared is the actual unchanged BF16 shared-expert output. All
    independent GEMMs use FP32 inputs/accumulation with TF32 forbidden.
    """
    _require(x.ndim == 2 and x.dtype == torch.bfloat16, "x must be BF16 [M,H]")
    _require(ids.ndim == 2 and ids.shape == weights.shape and ids.shape[0] == x.shape[0],
             "ids and weights must have matching [M,top_k] shapes")
    _require(ids.device == weights.device == x.device, "input devices must match")
    _require(ids.dtype in (torch.int32, torch.int64) and weights.dtype == torch.float32,
             "expected integer routes and FP32 route weights")
    _require(bool(torch.isfinite(x).all()) and bool(torch.isfinite(weights).all()),
             "nonfinite input")
    _require(swiglu_limit == 10., "only GLM SILU clamp=10 is qualified here")
    if x.is_cuda:
        _require(not torch.backends.cuda.matmul.allow_tf32, "disable TF32 for oracle GEMMs")
    e, two_i, half_h = raw["w13_weight"].shape
    _require(half_h * 2 == x.shape[1] and two_i % 2 == 0, "W13 shape mismatch")
    _require(tuple(raw["w2_weight"].shape) == (e, x.shape[1], two_i // 4),
             "W2 shape mismatch")
    _require(bool((ids >= 0).all()) and bool((ids < e).all()), "invalid expert ID")
    factors = {p: scale_factor(raw[p + "_weight_scale"]) for p in ("w13", "w2")}
    globals_ = {}
    cutoff = {}
    for p in factors:
        g = raw[p + "_weight_scale_2"]
        _require(g.device.type == "cpu" and g.dtype == torch.float32 and g.numel() == e,
                 "one raw FP32 global per expert required")
        _require(bool(torch.isfinite(g).all()) and bool((g >= 0).all()), "invalid global")
        globals_[p] = ((g.reshape(e) * (2. ** 119)) / factors[p]).to(x.device)
        _require(bool(torch.isfinite(globals_[p]).all()), "packed global overflow")
        s = raw[p + "_weight_scale"].float()
        cutoff[p] = int(((s > 0) & (s * factors[p] < 1 / 64)).sum())
    partials = torch.empty((*ids.shape, x.shape[1]), device=x.device, dtype=torch.bfloat16)
    hits = {"gate_above_10": 0, "up_outside_10": 0}
    for expert in range(e):
        rows, slots = torch.where(ids == expert)
        if rows.numel() == 0:
            continue
        w13 = decode_register_weight(raw["w13_weight"][expert],
            raw["w13_weight_scale"][expert], factors["w13"]).to(x.device)
        fc1 = (F.linear(x[rows].float(), w13.float()) * globals_["w13"][expert]).bfloat16()
        del w13
        h, clipped = silu_bf16(*fc1.chunk(2, -1), limit=swiglu_limit)
        for key in hits:
            hits[key] += clipped[key]
        w2 = decode_register_weight(raw["w2_weight"][expert],
            raw["w2_weight_scale"][expert], factors["w2"]).to(x.device)
        dot_bf16 = F.linear(h.float(), w2.float()).bfloat16()
        route_scale = (weights[rows, slots] * globals_["w2"][expert]).bfloat16()
        partials[rows, slots] = (dot_bf16.float() * route_scale[:, None].float()).bfloat16()
        del w2, fc1, h, dot_bf16
    routed = ordered_sum(partials)
    if shared is not None:
        _require(shared.shape == x.shape and shared.dtype == x.dtype and shared.device == x.device,
                 "shared output must match x device/shape/BF16")
        routed = (routed.float() + shared.float()).bfloat16()
    _require(bool(torch.isfinite(routed).all()), "nonfinite oracle output")
    return routed, {"oracle": "independent_literal_packed_bf16_v1", "b12x_commit": B12X_COMMIT,
                    "scale_factors": factors, "packing_cutoff_blocks": cutoff,
                    "clamp_hits": hits, "bitwise_kernel_emulator": False}


class CpuChecks(unittest.TestCase):
    def test_nibbles_and_scale_domain(self):
        packed = torch.tensor([list(range(0x10, 0x100, 0x22))], dtype=torch.uint8)
        self.assertEqual(packed.numel(), 8)
        w = decode_register_weight(packed, torch.ones(1, 1), 1)
        expected = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                                -0., -.5, -1., -1.5, -2., -3., -4., -6.])
        torch.testing.assert_close(w.float().reshape(-1) * 2.**119, expected, rtol=0, atol=0)

    def test_factor_is_projection_wide_and_cutoff_is_strict(self):
        self.assertEqual(scale_factor(torch.tensor([[[1.]], [[448.]]])), 1.)
        self.assertEqual(scale_factor(torch.tensor([[[1.]], [[2.]]])), 128.)
        q = torch.full((2, 8), 0x22, dtype=torch.uint8)
        w = decode_register_weight(q, torch.tensor([[1/128], [1/64]]), 1)
        self.assertEqual(int(torch.count_nonzero(w[0])), 0)
        self.assertEqual(int(torch.count_nonzero(w[1])), 16)

    def test_normalized_scale_identity_against_independent_register_bits(self):
        # Decode scalar IEEE bits independently of the LUT/scaling formula.
        # Exercise every positive finite E4M3 scale and every E2M1 code.
        source_scales = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
        for factor in (1., 2., 256.):
            s = source_scales[source_scales * factor <= 448]
            half = (s * factor * 128).half()
            half[half < 2] = 0
            byte = ((half.view(torch.int16).int() << 1) & 0xffff) >> 8
            scale_bits = (((byte & 128) << 7) | ((byte & 127) << 4)).short()
            scale_values = scale_bits.view(torch.bfloat16).float()
            for code in range(16):
                q_bits = torch.tensor([((code & 8) << 12) | ((code & 7) << 6)],
                                      dtype=torch.int32).short()
                q_value = q_bits.view(torch.bfloat16).float()
                expected = (q_value * scale_values).bfloat16()
                actual = decode_register_weight(
                    torch.full((s.numel(), 8), code | (code << 4), dtype=torch.uint8),
                    s[:, None], factor)
                self.assertTrue(torch.equal(actual[:, 0], expected))

    def test_silu_rounding_and_clipping(self):
        g = torch.tensor([[-20., .123, 20.]], dtype=torch.bfloat16)
        u = torch.tensor([[-20., 2.7, 20.]], dtype=torch.bfloat16)
        actual, hits = silu_bf16(g, u)
        gc = g.float().clamp(max=10)
        expected = ((gc / (1 + torch.exp(-gc))).bfloat16().float()
                    * u.float().clamp(-10, 10)).bfloat16()
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(hits, {"gate_above_10": 1, "up_outside_10": 2})

    def test_slot_order_is_observable(self):
        values = torch.tensor([[[2.**30], [-2.**30], [1.]]], dtype=torch.bfloat16)
        self.assertEqual(float(ordered_sum(values)), 1.)
        self.assertEqual(float(ordered_sum(values[:, [0, 2, 1]])), 0.)

    def test_tiny_full_oracle_and_invalid_route(self):
        raw = {}
        for p, shape in (("w13", (2, 32, 8)), ("w2", (2, 16, 8))):
            raw[p + "_weight"] = torch.full(shape, 0x22, dtype=torch.uint8)
            raw[p + "_weight_scale"] = torch.ones((*shape[:2], 1))
            raw[p + "_weight_scale_2"] = torch.tensor([.125, .25])
        x = torch.full((1, 16), .125, dtype=torch.bfloat16)
        ids = torch.tensor([[1, 0]])
        weights = torch.tensor([[.75, .25]])
        out, diagnostic = reference_packed(x, ids, weights, raw)
        gate = torch.tensor([[.5, .25]], dtype=torch.bfloat16)
        h, _ = silu_bf16(gate, gate)
        expected = (h.float() * 16 * torch.tensor([[.25, .125]]) * weights).bfloat16()
        expected = expected.float().sum(1).bfloat16()
        self.assertTrue(torch.equal(out, expected[:, None].expand_as(out)))
        self.assertEqual(diagnostic["scale_factors"], {"w13": 256., "w2": 256.})
        with self.assertRaisesRegex(ValueError, "invalid expert ID"):
            reference_packed(x, torch.tensor([[2, 0]]), weights, raw)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] != ["--self-test"]:
        raise SystemExit("usage: reference_packed.py --self-test")
    unittest.main(argv=[sys.argv[0]])
