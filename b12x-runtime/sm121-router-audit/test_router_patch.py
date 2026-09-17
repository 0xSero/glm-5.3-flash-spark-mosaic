import ast,pathlib,types,unittest
from stage_router_patch import patch
ROOT=pathlib.Path(__file__).parent
class PatchTest(unittest.TestCase):
 def test_unknown_source_rejected(self):
  with self.assertRaises(ValueError):patch(b'unknown')
 def test_actual_constructor_predicate_scope(self):
  tree=ast.parse(patch((ROOT/'gate_linear.installed.py').read_bytes()))
  cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GateLinear')
  init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
  # Execute actual initial eligibility assignments only; no kernel stubs/execution.
  before_super=[]
  for node in init.body:
   if isinstance(node,ast.If):break
   before_super.append(node)
  code=compile(ast.Module(body=before_super,type_ignores=[]),'eligibility','exec')
  for cap,h,e,bias,want_ll,want_special in [((12,1),4096,288,False,True,False),((12,1),4096,256,False,False,False),((12,1),4096,288,True,False,False),((12,0),4096,288,False,True,False),((10,0),7168,256,False,True,True),((8,9),4096,288,False,False,False)]:
   platform=types.SimpleNamespace(is_device_capability=lambda v,cap=cap:cap==v,is_device_capability_family=lambda v,cap=cap:cap[0]==v//10,is_cuda=lambda:True)
   obj=types.SimpleNamespace();ns={'current_platform':platform,'self':obj,'input_size':h,'output_size':e,'bias':bias}
   exec(code,ns)
   self.assertEqual(obj._can_use_ll_bf16,want_ll);self.assertEqual(ns['can_use_specialized_kernels'],want_special)
 def test_splitk_restricted_and_guard_retained(self):
  text=patch((ROOT/'gate_linear.installed.py').read_bytes()).decode()
  self.assertIn('not self._sm121_glm_router or x.shape[0] in (1, 2, 4)',text)
  self.assertIn('self._sm120_graph_pool_lifetime_guard = is_blackwell_rtx or self._sm121_glm_router',text)

class ForwardDispatchTest(unittest.TestCase):
 def test_measured_sizes_and_sm120_unchanged(self):
  tree=ast.parse(patch((ROOT/'gate_linear.installed.py').read_bytes()))
  cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GateLinear')
  forward=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
  condition=next(n for n in forward.body if isinstance(n,ast.If)).test
  code=compile(ast.Expression(condition),'actual-forward-condition','eval')
  for sm121,m,want in [(True,1,True),(True,2,True),(True,3,False),(True,4,True),(True,5,False),(True,16,False),(False,3,True),(False,16,True),(False,17,False)]:
   actual=eval(code,{'self':types.SimpleNamespace(allow_ll_bf16_gemm=True,_sm121_glm_router=sm121),'x':types.SimpleNamespace(shape=(m,4096),dtype='bf16'),'torch':types.SimpleNamespace(bfloat16='bf16')})
   self.assertEqual(actual,want)

if __name__=='__main__':unittest.main()
