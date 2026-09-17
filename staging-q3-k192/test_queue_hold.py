import importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('queue_quality',Path(__file__).with_name('queue_quality.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class HoldGuard(unittest.TestCase):
 def test_explicit_hold_blocks_idle_gpu_launch(self):
  with tempfile.TemporaryDirectory() as d:
   work=Path(d);(work/'HOLD_GPU_LAUNCH').touch()
   (work/'stage-receipt.json').write_text(json.dumps({'manifest_sha256':m.PIN}))
   disk=type('Disk',(),{'free':100*2**30})()
   with patch.object(m,'WORK',work),patch.object(m,'ROOT',work),patch.object(m,'cpu'),patch.object(m,'memory_available',return_value=100*2**30),patch.object(m.shutil,'disk_usage',return_value=disk),patch.object(m.subprocess,'check_output',side_effect=['','']) as run,patch.object(m.time,'sleep',side_effect=InterruptedError('test stop')):
    with self.assertRaises(InterruptedError):m.execute()
    status=json.loads((work/'queue-status.json').read_text())
    self.assertTrue(status['explicit_gpu_hold'])
    self.assertEqual(status['state'],'WAITING_RESOURCE_ADMISSION')
    self.assertEqual(run.call_count,2)
    self.assertFalse((work/'launch.json').exists())
if __name__=='__main__':unittest.main()
