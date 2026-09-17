"""CPU-only launch and actual upstream dynamic-schedule contract checks."""
import importlib.util
import unittest
from pathlib import Path
from launch_controls import controls

UPSTREAM = Path(__file__).parent.parent / "b12x-runtime/upstream-vllm/vllm"


class LaunchControlsTest(unittest.TestCase):
    def test_all_fixed_points_cover_transient_active_batches(self):
        for depth in (1, 3, 5):
            for slots in (1, 2, 4, 8):
                for chunk in (2048, 4096, 8192):
                    c = controls(depth, slots, chunk)
                    sizes = c["compilation_config"]["cudagraph_capture_sizes"]
                    self.assertEqual(sizes[-1], (depth + 1) * slots)
                    self.assertEqual({n // (depth + 1) for n in sizes}, set(range(1, slots + 1)))
                    self.assertLessEqual(sizes[-1], chunk)
                    self.assertEqual(c["max_model_len"], 262144)
                    self.assertEqual(c["gpu_admission"], "UNTESTED")

    def test_out_of_scope_values_fail_closed(self):
        for bad in ((0, 1, 2048), (3, 16, 2048), (3, 8, 512)):
            with self.assertRaises(ValueError):
                controls(*bad)

    def test_actual_jovian_dynamic_schedule(self):
        path = UPSTREAM / "v1/spec_decode/dynamic/utils.py"
        spec = importlib.util.spec_from_file_location("dynamic_schedule_actual_source", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        schedule = [[1, 1, 5], [2, 2, 3], [3, 8, 1]]
        self.assertEqual(module.build_dynamic_sd_schedule_lookup(schedule, 8, 5), [0, 5, 3, 1, 1, 1, 1, 1, 1])
        with self.assertRaises(ValueError):
            module.build_dynamic_sd_schedule_lookup([[2, 8, 1]], 8, 5)


if __name__ == "__main__":
    unittest.main()
