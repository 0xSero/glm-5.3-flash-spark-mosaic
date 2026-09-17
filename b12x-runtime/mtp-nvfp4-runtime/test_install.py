"""CPU source-only installer regression tests using the pinned source closure."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from install import install

SOURCE = Path(__file__).parent
PARENT = SOURCE.parent
PINS = json.loads((SOURCE / "SOURCE_PINS.json").read_text())


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=SOURCE)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.v = self.root / "vllm"
        self.b = PARENT / "upstream-b12x"
        snapshots = json.loads((SOURCE / "dispatch-source.json").read_text())
        for name in PINS["vllm_sources"]:
            dst = self.v / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if name in snapshots:
                dst.write_text(snapshots[name])
            elif "glm_mtp_expert_fp8" in name:
                shutil.copyfile(PARENT / "mtp_expert_fp8.py", dst)
            else:
                shutil.copyfile(PARENT / "upstream-vllm/vllm" / name, dst)

    def test_install_and_exact_idempotence(self):
        first = install(self.v, self.b, SOURCE)
        self.assertEqual(first, install(self.v, self.b, SOURCE))
        for name, expected in PINS["dispatch_after"].items():
            self.assertEqual(hashlib.sha256((self.v / name).read_bytes()).hexdigest(), expected)

    def test_check_only_does_not_write(self):
        result = install(self.v, self.b, SOURCE, True)
        self.assertEqual(result["state"], "CHECKED")
        self.assertFalse((self.v / "model_executor/layers/quantization/glm_mtp_expert_precision.py").exists())
        for name, expected in PINS["vllm_sources"].items():
            self.assertEqual(hashlib.sha256((self.v / name).read_bytes()).hexdigest(), expected)

    def test_unknown_source_rejected_before_any_write(self):
        other = self.v / "v1/spec_decode/llm_base_proposer.py"
        other.write_text(other.read_text() + "\n# unknown local modification\n")
        unchanged = (self.v / "v1/worker/gpu/spec_decode/eagle/utils.py").read_bytes()
        with self.assertRaises(ValueError):
            install(self.v, self.b, SOURCE)
        self.assertEqual(unchanged, (self.v / "v1/worker/gpu/spec_decode/eagle/utils.py").read_bytes())


if __name__ == "__main__":
    unittest.main()
