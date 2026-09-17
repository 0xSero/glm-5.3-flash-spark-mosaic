"""Preserve compiler header precedence and cached unchanged Torch headers."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, required=True)
args = parser.parse_args()
cmake = args.root / 'CMakeLists.txt'
text = cmake.read_text()
old = '''      file(MAKE_DIRECTORY "${TORCH_SHADOW_INCLUDE_DIR}/torch/csrc/stable")
      file(COPY_FILE
        "${TORCH_INCLUDE_DIR}/${_stable_hdr_rel}"
        "${TORCH_SHADOW_INCLUDE_DIR}/${_stable_hdr_rel}")

      execute_process(
        COMMAND patch --batch --forward -p1
          "--input=${CMAKE_CURRENT_LIST_DIR}/cmake/patches/pytorch_stable_string.patch"
        WORKING_DIRECTORY "${TORCH_SHADOW_INCLUDE_DIR}"
        COMMAND_ERROR_IS_FATAL ANY)
'''
new = '''      # Patch a temporary copy, preserving unchanged compiled-header mtimes.
      set(TORCH_SHADOW_PATCH_DIR "${TORCH_SHADOW_INCLUDE_DIR}/.patch-staging")
      file(MAKE_DIRECTORY "${TORCH_SHADOW_PATCH_DIR}/torch/csrc/stable")
      file(MAKE_DIRECTORY "${TORCH_SHADOW_INCLUDE_DIR}/torch/csrc/stable")
      file(COPY_FILE
        "${TORCH_INCLUDE_DIR}/${_stable_hdr_rel}"
        "${TORCH_SHADOW_PATCH_DIR}/${_stable_hdr_rel}")

      execute_process(
        COMMAND patch --batch --forward -p1
          "--input=${CMAKE_CURRENT_LIST_DIR}/cmake/patches/pytorch_stable_string.patch"
        WORKING_DIRECTORY "${TORCH_SHADOW_PATCH_DIR}"
        COMMAND_ERROR_IS_FATAL ANY)
      file(COPY_FILE
        "${TORCH_SHADOW_PATCH_DIR}/${_stable_hdr_rel}"
        "${TORCH_SHADOW_INCLUDE_DIR}/${_stable_hdr_rel}" ONLY_IF_DIFFERENT)
      file(REMOVE_RECURSE "${TORCH_SHADOW_PATCH_DIR}")
'''
if new not in text:
    if text.count(old) != 1:
        raise ValueError('Pinned Torch header patch anchor changed')
    cmake.write_text(text.replace(old, new))

helper = args.root / 'tools/build_deepgemm_C.py'
text = helper.read_text()
old = '    *(f"-I{p}" for p in includes),\n'
new = '''    *(f"-I{p}" for p in includes),
    # Vendor math headers are a fallback after nvcc's own CUDA/CRT headers.
    "-idirafter", str(Path(torch.__file__).parent.parent / "nvidia/cu13/include"),
'''
if new not in text:
    if text.count(old) != 1:
        raise ValueError('Pinned DeepGEMM compiler anchor changed')
    helper.write_text(text.replace(old, new))
print(json.dumps({'state': 'BUILD_PATH_REPAIRS_APPLIED', 'sha256': {
    str(p.relative_to(args.root)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (cmake, helper)}}))
