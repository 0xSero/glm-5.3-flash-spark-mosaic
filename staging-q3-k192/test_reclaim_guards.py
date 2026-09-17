import importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('stage',Path(__file__).with_name('reclaim_and_stage.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class ReclaimGuards(unittest.TestCase):
 def test_unverified_backup_never_deletes(self):
  with tempfile.TemporaryDirectory() as d:
   work=Path(d);(work/'verify_preserved_copy.py').write_text('read only')
   with patch.object(m,'WORK',work),patch.object(m,'status'),patch.object(m.subprocess,'check_output',return_value=b'{"state":"FAILED"}'),patch.object(Path,'unlink') as delete:
    with self.assertRaises(AssertionError):m.execute()
    delete.assert_not_called()
 def test_different_backup_pin_never_deletes(self):
  with tempfile.TemporaryDirectory() as d:
   work=Path(d);(work/'verify_preserved_copy.py').write_text('read only')
   proof=json.dumps({'state':'PRESERVED_COPY_FULL_HASH_PASS','manifest_sha256':'other'}).encode()
   with patch.object(m,'WORK',work),patch.object(m,'status'),patch.object(m.subprocess,'check_output',return_value=proof),patch.object(Path,'unlink') as delete:
    with self.assertRaises(AssertionError):m.execute()
    delete.assert_not_called()
if __name__=='__main__':unittest.main()
