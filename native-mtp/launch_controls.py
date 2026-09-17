"""Emit experimental launch controls; never launch or claim GPU admission."""
import argparse
import json


def controls(depth, slots, chunk):
    if depth not in (1, 3, 5) or slots not in (1, 2, 4, 8) or chunk not in (2048, 4096, 8192):
        raise ValueError("Use audited depth 1/3/5, slots 1/2/4/8, chunk 2048/4096/8192")
    # V2 ModelState contributes one sampled token per target decode step.
    # Capture every active request count, including transients between sweep points.
    sizes = [(depth + 1) * n for n in range(1, slots + 1)]
    return {
        "max_model_len": 262144,
        "max_num_seqs": slots,
        "max_num_batched_tokens": chunk,
        "speculative_config": {"method": "mtp", "model": "/mtp", "num_speculative_tokens": depth},
        "compilation_config": {
            "cudagraph_mode": "FULL_DECODE_ONLY", "custom_ops": ["all"],
            "cudagraph_capture_sizes": sizes, "max_cudagraph_capture_size": sizes[-1],
        },
        "gpu_admission": "UNTESTED",
        "scope": "Startup config only; KV capacity and active concurrency require runtime evidence",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--depth", type=int, choices=(1, 3, 5), required=True)
    p.add_argument("--slots", type=int, choices=(1, 2, 4, 8), required=True)
    p.add_argument("--chunk", type=int, choices=(2048, 4096, 8192), required=True)
    a = p.parse_args()
    print(json.dumps(controls(a.depth, a.slots, a.chunk), indent=2))
