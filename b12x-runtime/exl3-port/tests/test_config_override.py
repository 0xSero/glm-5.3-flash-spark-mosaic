import importlib.util,tempfile,unittest,ast
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('config_order_patch',HERE/'apply_config_override.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
SOURCE=HERE.parent/'upstream-vllm/vllm/config/model.py'
class ConfigOverrideTests(unittest.TestCase):
 def test_idempotent_exact_pinned_source(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'model.py';p.write_bytes(SOURCE.read_bytes())
   first=m.apply(p);self.assertEqual(first['after_sha256'],m.PATCHED_SHA)
   second=m.apply(p);self.assertEqual(second['state'],'CONFIG_OVERRIDE_ALREADY_APPLIED')
   tree=ast.parse(p.read_text());fn=next(n for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='ModelConfig' for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='_verify_quantization')
   lists=[n.value for n in ast.walk(fn) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='overrides' for t in n.targets) and isinstance(n.value,ast.List)]
   self.assertEqual(len(lists),1);values=ast.literal_eval(lists[0]);self.assertEqual(values.count('exl3'),1);self.assertEqual(values[0],'exl3')
 def test_unknown_source_untouched(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'model.py';data=SOURCE.read_bytes()+b'\n# unrelated change\n';p.write_bytes(data)
   with self.assertRaisesRegex(ValueError,'differs'):m.apply(p)
   self.assertEqual(p.read_bytes(),data)
 def test_check_only_does_not_mutate(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'model.py';data=SOURCE.read_bytes();p.write_bytes(data)
   self.assertEqual(m.apply(p,True)['state'],'CONFIG_OVERRIDE_CHECKED');self.assertEqual(p.read_bytes(),data)
if __name__=='__main__':unittest.main()
