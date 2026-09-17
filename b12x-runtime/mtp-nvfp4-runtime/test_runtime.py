"""Actual imports; constructor/platform/planner mocks are explicit. No GPU work."""
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
INSTALLED = os.environ.get("GLM53_TEST_INSTALLED_POLICY") == "1"
if INSTALLED:
    from vllm.model_executor.layers.quantization import glm_mtp_expert_precision as p
else:
    import glm_mtp_expert_precision as p

FLAGS = {"GLM53_MTP_EXPERT_FP8": "0", "GLM53_MTP_EXPERT_NVFP4": "1",
         "VLLM_MTP_NVFP4_LM_HEAD": "0", "VLLM_MXFP8_LM_HEAD": "0",
         "VLLM_B12X_MOE_FP4_FORCE_A16": "1"}


@dataclass
class Config:
    model_config: object
    quant_config: object = None
    speculative_config: object = None


def native(experts=288, arch="Glm5NextMTPModel", quant=None):
    return Config(NS(hf_config=NS(architectures=[arch], n_routed_experts=experts)), quant)


@dataclass
class Geometry:
    hidden_dim: int = 4096
    intermediate_size: int = 2048
    tp_size: int = 1
    num_experts: int = 288
    experts_per_token: int = 8
    in_dtype: object = torch.bfloat16
    has_bias: bool = False
    moe_backend: str = "auto"
    moe_parallel_config: object = None


@dataclass(frozen=True)
class FakePolicy:
    backend: str = "w4a16"
    w4a16_route_mode: str = "direct"


@dataclass(frozen=True)
class FakeContext:
    config: object = FakePolicy()
    def with_override(self, key, config):
        assert key == "moe.decode"
        return FakeContext(config)


@dataclass(frozen=True)
class FakeCaps:
    max_tokens: int
    num_topk: int
    device: object
    weight_plan: object
    core_token_counts: tuple
    route_num_experts: int
    quant_mode: str
    apply_router_weight_on_input: bool
    swiglu_limit: float
    swiglu_alpha: float
    swiglu_beta: float
    frozen: bool
    w4a16_fast_math: bool
    policy_context: object = FakeContext()


class Tests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, FLAGS))

    def test_flags_and_separate_native_config(self):
        c = native()
        q = p.maybe_enable_mtp_expert_precision(c)
        self.assertIsNone(c.quant_config)
        self.assertIsInstance(q.quant_config, p.MTPExpertOnlyNvfp4Config)
        for bad in (q, native(256), native(arch="Glm5NextForConditionalGeneration")):
            with self.assertRaises(ValueError):
                p.maybe_enable_mtp_expert_precision(bad)
        for flag in FLAGS:
            bad_value = "1" if flag in ("GLM53_MTP_EXPERT_FP8", "VLLM_MTP_NVFP4_LM_HEAD",
                                       "VLLM_MXFP8_LM_HEAD") else "invalid"
            with patch.dict(os.environ, {flag: bad_value}), self.assertRaises(ValueError):
                p.maybe_enable_mtp_expert_precision(c)
        for flag in ("VLLM_MTP_NVFP4_LM_HEAD", "VLLM_MXFP8_LM_HEAD"):
            with patch.dict(os.environ):
                del os.environ[flag]
                with self.assertRaises(ValueError):
                    p.maybe_enable_mtp_expert_precision(c)

    def test_defaults_and_existing_fp8_policy(self):
        c = native()
        with patch.dict(os.environ, {"GLM53_MTP_EXPERT_NVFP4": "0"}):
            self.assertIs(p.maybe_enable_mtp_expert_precision(c), c)
            with patch.dict(os.environ, {"GLM53_MTP_EXPERT_FP8": "1"}):
                self.assertEqual(type(p.maybe_enable_mtp_expert_precision(c).quant_config).__name__,
                                 "MTPExpertOnlyFp8Config")

    @unittest.skipUnless(INSTALLED, "needs installed dispatch overlay")
    def test_actual_v2_dispatch_rebinds_before_model_load(self):
        # Execute the installed dispatcher up to the get_model boundary only.
        # Backend setup and the loader are mocked, never any model weights.
        from vllm.v1.worker.gpu.spec_decode.eagle import utils
        from vllm.model_executor.models import utils as model_utils
        draft = native().model_config
        c = native(arch="Glm5NextForConditionalGeneration", quant=object())
        original_quant = c.quant_config
        c.speculative_config = NS(draft_model_config=draft)
        class LoadBoundary(Exception):
            pass
        with patch.object(utils, "_make_eagle_draft_vllm_config", side_effect=lambda c: c), \
                patch.object(model_utils, "get_draft_quant_config", return_value=None), \
                patch("vllm.compilation.backends.set_model_tag", return_value=nullcontext()), \
                patch.object(utils, "get_model", side_effect=LoadBoundary) as loader:
            with self.assertRaises(LoadBoundary):
                utils.load_eagle_model(NS(), c)
        bound = loader.call_args.kwargs
        self.assertIs(bound["model_config"], draft)
        self.assertIs(bound["vllm_config"].model_config, draft)
        self.assertIsInstance(bound["vllm_config"].quant_config, p.MTPExpertOnlyNvfp4Config)
        self.assertIs(c.quant_config, original_quant)

    def test_protected_and_target_scope(self):
        class Dense(p.LinearBase):
            def __init__(self):
                torch.nn.Module.__init__(self)
            def forward(self, *args):
                raise AssertionError("CPU test must not run a kernel")
        q = p.MTPExpertOnlyNvfp4Config()
        for prefix in ("model.layers.45.mlp.shared_experts.down_proj", "lm_head",
                       "model.layers.45.self_attn.q_proj"):
            self.assertIsInstance(q.get_quant_method(Dense(), prefix), p.UnquantizedLinearMethod)
        self.assertIsNone(q.get_quant_method(torch.nn.Embedding(2, 2), "model.embed_tokens"))
        e = p.RoutedExperts.__new__(p.RoutedExperts)
        torch.nn.Module.__init__(e)
        with self.assertRaises(ValueError):
            q.get_quant_method(e, "model.layers.44.mlp.experts")
        with patch.object(p, "B12xNvfp4DraftMethod") as method:
            self.assertIs(q.get_quant_method(e, "model.layers.45.mlp.experts"), method.return_value)
            method.assert_called_once_with(layer=e)

    def test_geometry_and_exact_backend_selection(self):
        g = Geometry(moe_parallel_config=NS(ep_size=1, dp_size=1, enable_eplb=False))
        def initialize(method, selected):
            method.moe = selected
        with patch.object(p, "current_platform") as platform, \
                patch.object(p.OnlineMoEMethodBase, "__init__", initialize), \
                patch.object(p, "select_nvfp4_moe_backend",
                    return_value=(p.NvFp4MoeBackend.B12X, p.B12xExperts)) as select:
            platform.is_cuda.return_value = platform.is_device_capability.return_value = True
            method = p.B12xNvfp4DraftMethod(layer=NS(moe_config=g))
            self.assertEqual(g.moe_backend, "auto")
            self.assertEqual(method.moe.moe_backend, "b12x")
            self.assertIsNone(select.call_args.kwargs["activation_key"])
            self.assertIs(method.experts_cls, p.PackedDraftB12xExperts)
            for bad in (replace(g, num_experts=8), replace(g, tp_size=2), replace(g, has_bias=True),
                        replace(g, in_dtype=torch.float16),
                        replace(g, moe_parallel_config=NS(ep_size=1, dp_size=1, enable_eplb=True))):
                with self.assertRaises(ValueError):
                    p.B12xNvfp4DraftMethod(layer=NS(moe_config=bad))
            select.return_value = (p.NvFp4MoeBackend.B12X, object)
            with self.assertRaises(ValueError):
                p.B12xNvfp4DraftMethod(layer=NS(moe_config=g))

    def fake_kernel(self):
        k = p.PackedDraftB12xExperts.__new__(p.PackedDraftB12xExperts)
        k._plans = {}
        k._packed_warmed_plans = {}
        k._prefill_capacity = 2048
        k._plan_capacities = {1, 2, 4, 8, 2048}
        k._quant_mode = "w4a16"
        k.moe_config = NS(experts_per_token=8, in_dtype=torch.bfloat16)
        k._apply_router_weight_on_input = False
        k._prepared_experts = NS(plan=NS(source_format="modelopt_nvfp4"), w1_fp4=NS(device="cpu"))
        k._swiglu_params = lambda activation: (10., 1., 0.)
        plans = []
        def factory(caps):
            plan = NS(caps=caps, layout=NS(total_nbytes=512),
                launch_plan=NS(policy_resolution=NS(config=caps.policy_context.config)))
            plans.append(plan)
            return plan
        capturing = self.stack.enter_context(patch.object(p, "_is_current_stream_capturing", return_value=False))
        self.stack.enter_context(patch.object(p, "_require_b12x_fused_moe", return_value=NS(Caps=FakeCaps, plan=factory)))
        return k, plans, capturing

    def test_packed_cache_and_context_isolation(self):
        k, plans, _ = self.fake_kernel()
        a = k._plan(tokens=1, topk=8, activation=MoEActivation.SILU)
        self.assertEqual(len(plans), 2)
        self.assertEqual(plans[0].caps.policy_context.config.w4a16_route_mode, "direct")
        self.assertEqual(a.caps.policy_context.config.w4a16_route_mode, "packed")
        self.assertIs(a.caps.weight_plan, plans[0].caps.weight_plan)
        self.assertIs(k._plan(tokens=1, topk=8, activation=MoEActivation.SILU), a)
        self.assertEqual(len(plans), 2)
        self.assertIs(type(k).workspace_shapes, p.B12xExperts.workspace_shapes)
        self.assertIs(type(k).apply, p.B12xExperts.apply)
        with patch("vllm.model_executor.layers.fused_moe.b12x._b12x_scratch_nbytes", return_value=513):
            self.assertEqual(k.workspace_shapes(1, 0, 4096, 8, 288, 288, None, MoEActivation.SILU),
                             ((0,), (257,), (1, 4096)))

    def test_capture_requires_completed_matching_warmup(self):
        k, plans, capture = self.fake_kernel()
        capture.return_value = True
        with self.assertRaisesRegex(RuntimeError, "created before"):
            k._plan(tokens=1, topk=8, activation=MoEActivation.SILU)
        capture.return_value = False
        first = k._plan(tokens=1, topk=8, activation=MoEActivation.SILU)
        capture.return_value = True
        with self.assertRaisesRegex(RuntimeError, "completed eager warmup"):
            k._plan(tokens=1, topk=8, activation=MoEActivation.SILU)
        capture.return_value = False
        layer = NS(activation=MoEActivation.SILU, apply_router_weight_on_input=False)
        warmed = []
        def fake_warmup(self, layer, *, token_counts):
            warmed.append(token_counts)
            self._plan(tokens=token_counts[0], topk=8, activation=layer.activation)
            return 1
        with patch.object(p.B12xExperts, "warmup_launches", fake_warmup):
            self.assertEqual(k.warmup_launches(layer, token_counts=(2, 1, 2)), 2)
        self.assertEqual(warmed, [(1,), (2,)])
        metadata = k.packed_plan_metadata()
        self.assertEqual([row["capacity"] for row in metadata], [1, 2])
        self.assertTrue(all(row["route_mode"] == "packed" and row["eager_warmup_completed"]
                            and row["fast_math"] for row in metadata))
        capture.return_value = True
        self.assertIs(k._plan(tokens=1, topk=8, activation=MoEActivation.SILU), first)
        with self.assertRaises(RuntimeError):
            k.warmup_launches(layer, token_counts=(4,))

    def test_failed_replanning_never_caches_direct_plan(self):
        k, _, _ = self.fake_kernel()
        with patch.object(p, "_require_b12x_fused_moe") as lib:
            lib.return_value.Caps = FakeCaps
            lib.return_value.plan.side_effect = RuntimeError("planning failed")
            with self.assertRaises(RuntimeError):
                k._plan(tokens=1, topk=8, activation=MoEActivation.SILU)
        self.assertEqual(k._plans, {})


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert not torch.cuda.is_initialized(), "CPU gate initialized CUDA"
    print(json.dumps({"state": "CPU_PASS" if result.wasSuccessful() else "CPU_FAILED",
        "tests": result.testsRun, "cuda_initialized": False,
        "actual_imports": True, "GPU_constructor_and_plan_factory_mocked": True,
        "installed_policy": INSTALLED,
        "full_engine_config_or_serving_proven": False}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
