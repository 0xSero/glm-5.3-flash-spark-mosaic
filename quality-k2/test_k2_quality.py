import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from validate_source import check_trellis, validate_metadata, validate_file, INVENTORY_SHA, sha

ROOT=Path(__file__).parent
BUNDLE=ROOT.parent/'source-inventory-k2'


class K2Tests(unittest.TestCase):
    def test_pinned_metadata_and_protected_q3_parity(self):
        self.assertEqual(sha(BUNDLE/'filehash-inventory.json'),INVENTORY_SHA)
        inv=json.loads((BUNDLE/'filehash-inventory.json').read_text())
        manifest=validate_metadata(BUNDLE,inv,BUNDLE)
        self.assertEqual(manifest['effective_routed_tier_bpw'],2.0)
        proof=json.loads((BUNDLE/'protected-q3-parity.json').read_text())
        self.assertEqual(len(proof['files']),49)
        self.assertTrue(proof['all_retained_shards_identical'])

    def test_runtime_config_override_rejected_without_mutation(self):
        inv=json.loads((BUNDLE/'filehash-inventory.json').read_text())
        original=sha(BUNDLE/'config.json')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in inv['metadata_sha256']:
                if name!='protected-q3-parity.json':(root/name).symlink_to(BUNDLE/name)
            (root/'config.json').unlink();(root/'config.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'local metadata differs'):
                validate_metadata(root,inv,BUNDLE)
        self.assertEqual(sha(BUNDLE/'config.json'),original)

    def test_header_rejects_k3_and_wrong_orientation(self):
        down='model.language_model.layers.3.mlp.experts.287.down_proj.rank3.trellis'
        gate=down.replace('down_proj','gate_proj')
        self.assertEqual(check_trellis(down,{'dtype':'I16','shape':[32,256,32],'data_offsets':[0,524288]}),1)
        self.assertEqual(check_trellis(gate,{'dtype':'I16','shape':[256,32,32],'data_offsets':[0,524288]}),1)
        with self.assertRaises(ValueError):check_trellis(down,{'dtype':'I16','shape':[32,256,48],'data_offsets':[0,786432]})
        with self.assertRaises(ValueError):check_trellis(gate,{'dtype':'I16','shape':[32,256,32],'data_offsets':[0,524288]})

    def test_size_only_is_not_payload_verification(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'weights';p.write_bytes(b'abcd')
            item={'path':'weights','bytes':4,'sha256':hashlib.sha256(b'abcd').hexdigest()}
            validate_file(root,item)
            p.write_bytes(b'abce')
            validate_file(root,item,full_hash=False)
            with self.assertRaises(ValueError):validate_file(root,item)

    def test_help_does_not_require_gpu_imports(self):
        for name in ['validate_source.py','evaluate_k2_quality.py']:
            result=subprocess.run([sys.executable,str(ROOT/name),'--help'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
