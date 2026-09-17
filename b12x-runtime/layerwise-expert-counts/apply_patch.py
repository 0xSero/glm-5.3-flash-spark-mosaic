"""Apply only to the exact pinned Jovian GLM model or this known patched state."""
import argparse
import hashlib
from pathlib import Path

ORIGINAL_SHA = "5c86beadb1d14b648fb6244160997afba9b8d9da031fc0f232b32d5760ecbbbf"
HERE = Path(__file__).resolve().parent


def patched_source(original):
    if hashlib.sha256(original).hexdigest() != ORIGINAL_SHA:
        raise ValueError("Unknown Jovian source; refusing patch")
    text = original.decode()
    old = "        parallel_config = vllm_config.parallel_config\n\n        self.hidden_size = config.hidden_size\n"
    new = "        parallel_config = vllm_config.parallel_config\n        from .layerwise_counts import layer_config\n\n        config = layer_config(config, parallel_config, layer_idx, is_mtp_layer)\n\n        self.hidden_size = config.hidden_size\n"
    assert text.count(old) == 1
    text = text.replace(old, new)
    anchor = "        pending_attn_weights: dict = {}\n\n        for args in weights:\n"
    replacement = "        pending_attn_weights: dict = {}\n        from .layerwise_counts import plan, validate_loaded_tensor\n\n        layer_expert_counts = plan(self.config)\n        for args in weights:\n"
    assert text.count(anchor) == 1
    text = text.replace(anchor, replacement)
    old = "            name, loaded_weight = args[:2]\n            kwargs: dict = args[2] if len(args) > 2 else {}\n"
    new = "            name, loaded_weight = args[:2]\n            validate_loaded_tensor(self.config, name, loaded_weight.shape, layer_expert_counts)\n            kwargs: dict = args[2] if len(args) > 2 else {}\n"
    assert text.count(old) == 1
    return text.replace(old, new).encode()


def apply(path, check_only=False):
    original = (HERE / "model.original.py").read_bytes()
    patched = patched_source(original)
    if path.read_bytes() not in (original, patched):
        raise ValueError("Unknown target bytes; refusing mutation")
    helper = path.parent / "layerwise_counts.py"
    expected_helper = (HERE / "layerwise_counts.py").read_bytes()
    if helper.exists() and helper.read_bytes() != expected_helper:
        raise ValueError("Unknown existing helper; refusing overwrite")
    if not check_only:
        helper.write_bytes(expected_helper)
        path.write_bytes(patched)
    return hashlib.sha256(patched).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    print(apply(args.model_file, args.check_only))
