"""Install into a NEW image only; strict source pins, staged all-or-nothing checks."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def install(vllm_root, b12x_root, source, check_only=False):
    pins = json.loads((source / "SOURCE_PINS.json").read_text())
    policy = (source / "glm_mtp_expert_precision.py").read_bytes()
    if digest(policy) != pins["policy_sha256"]:
        raise ValueError("Runtime policy changed after sealing")
    staged = {}
    for name, expected in pins["vllm_sources"].items():
        path = vllm_root / name
        before = path.read_bytes()
        if name in pins["dispatch_after"]:
            if digest(before) == pins["dispatch_after"][name]:
                after = before
            elif digest(before) == expected:
                text = before.decode()
                if text.count("quantization.glm_mtp_expert_fp8 import") != 1:
                    raise ValueError("Native draft import anchor changed")
                if text.count("maybe_enable_mtp_expert_fp8") != 2:
                    raise ValueError("Native draft function anchor changed")
                after = text.replace("quantization.glm_mtp_expert_fp8 import",
                    "quantization.glm_mtp_expert_precision import").replace(
                    "maybe_enable_mtp_expert_fp8", "maybe_enable_mtp_expert_precision").encode()
            else:
                raise ValueError("Unknown native draft dispatcher: " + name)
            if digest(after) != pins["dispatch_after"][name]:
                raise ValueError("Dispatch patch hash mismatch")
            staged[path] = after
        elif digest(before) != expected:
            raise ValueError("Pinned vLLM dependency changed: " + name)
    for name, expected in pins["b12x_sources"].items():
        if digest((b12x_root / name).read_bytes()) != expected:
            raise ValueError("Pinned B12x dependency changed: " + name)
    target = vllm_root / "model_executor/layers/quantization/glm_mtp_expert_precision.py"
    if target.exists() and target.read_bytes() != policy:
        raise ValueError("Conflicting runtime policy already installed")
    staged[target] = policy
    for path, data in staged.items():
        compile(data, str(path), "exec")
    if not check_only:
        for path, data in staged.items():
            temporary = path.with_suffix(path.suffix + ".glm-nvfp4-tmp")
            temporary.write_bytes(data)
            temporary.replace(path)
    return {"state": "CHECKED" if check_only else "INSTALLED_SERVING_UNTESTED",
            "sha256": {str(p.relative_to(vllm_root)): digest(b) for p, b in staged.items()},
            "cuda_or_model_weights_loaded": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--b12x-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path(__file__).parent)
    parser.add_argument("--check-only", action="store_true")
    a = parser.parse_args()
    print(json.dumps(install(a.vllm_root, a.b12x_root, a.source, a.check_only), indent=2))
