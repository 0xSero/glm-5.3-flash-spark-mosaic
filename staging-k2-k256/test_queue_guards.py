import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('queue_quality',Path(__file__).with_name('queue_quality.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class QueueGuards(unittest.TestCase):
 def setup_queue(self,path):
  (path/'stage-receipt.json').write_text(json.dumps({'state':'FULL_PAYLOAD_HASH_PASS','candidate_manifest_sha256':m.PIN}))
 def test_running_baseline_cannot_launch(self):
  with tempfile.TemporaryDirectory() as d:
   q=Path(d);self.setup_queue(q)
   old={'Id':m.BASELINE,'Image':m.IMAGE,'State':{'Running':True}}
   with patch.object(m,'QUEUE',q),patch.object(m,'inspect',return_value=old),patch.object(m,'status'),patch.object(m.time,'sleep',side_effect=RuntimeError('waiting')),patch.object(m.subprocess,'check_output') as launch:
    with self.assertRaisesRegex(RuntimeError,'waiting'):m.main()
    launch.assert_not_called()
 def test_failed_baseline_cannot_launch(self):
  with tempfile.TemporaryDirectory() as d:
   q=Path(d);self.setup_queue(q)
   old={'Id':m.BASELINE,'Image':m.IMAGE,'State':{'Running':False,'ExitCode':1}}
   with patch.object(m,'QUEUE',q),patch.object(m,'inspect',return_value=old),patch.object(m.subprocess,'run'),patch.object(m.subprocess,'check_output') as launch:
    with self.assertRaisesRegex(RuntimeError,'original Q3 control failed'):m.main()
    launch.assert_not_called()

if __name__=='__main__':unittest.main()
