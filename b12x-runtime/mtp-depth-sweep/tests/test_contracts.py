import copy,importlib.util,json,os,pathlib,subprocess,sys,tempfile,types,unittest
from unittest.mock import patch
HERE=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import admission,counters,launch,measure,capture_plugin,source_contract


def sample(depth=3,engine='0',count=1,created=123):
 labels=f'model_name="glm",engine="{engine}",worker="same"'
 values=[(counters.BASES[0],10),(counters.BASES[1],27),(counters.BASES[2],18)]
 rows=[f'{name}_total{{{labels}}} {value*count}' for name,value in values]
 rows += [f'{counters.BASES[3]}_total{{{labels},position="{i}"}} {value*count}' for i,value in enumerate([9,6,3][:depth])]
 rows += [f'{counters.BASES[0]}_created{{{labels}}} {created}']
 return '\n'.join(rows)

def snap(text):return {'model':'glm','depth':3,'identity':{'pid':1},'series':counters.parse(text,'glm',3)}

def record(cls,sizes,q,profile=False):
 return {'manager_class':cls,'capture_completed':True,'profile_sample':profile,'descriptors':[{'cg_mode':'FULL','num_active_loras':0,'num_reqs':n,'num_tokens':n*q,'uniform_token_count':q} for n in sizes]}

class Counters(unittest.TestCase):
 def test_positions_and_engine_labels_preserved(self):
  a=snap(sample()+ '\n'+sample(engine='1'));b=snap(sample(count=2)+'\n'+sample(engine='1',count=2))
  result=counters.compare(a,b)
  self.assertEqual(len(result['engines']),2);self.assertEqual(result['accepted_tokens'],36)
  self.assertEqual(result['engines'][0]['accepted_tokens_by_position'],{0:9,1:6,2:3})
  self.assertEqual(result['mean_acceptance_length'],2.8)
  self.assertEqual(result['engines'][1]['labels']['worker'],'same')
 def test_unrelated_model_filtered(self):
  self.assertEqual(counters.parse(sample()+'\n'+sample().replace('"glm"','"other"'),'glm',3),counters.parse(sample(),'glm',3))
 def test_escaped_labels(self):
  self.assertEqual(counters.labels('model_name="a\\\"b",engine="0"')['model_name'],'a"b')
 def test_bad_labels(self):
  for text in ('x="y",','x="y",x="z"','x="y" z="a"'):
   with self.subTest(text=text),self.assertRaises(ValueError):counters.labels(text)
 def test_missing_position_rejected(self):
  with self.assertRaisesRegex(ValueError,'coverage'):counters.parse('\n'.join(x for x in sample().splitlines() if 'position="1"' not in x),'glm',3)
 def test_missing_engine(self):
  with self.assertRaisesRegex(ValueError,'engine'):counters.parse(sample().replace(',engine="0"',''),'glm',3)
 def test_duplicate_series(self):
  with self.assertRaisesRegex(ValueError,'Duplicate'):counters.parse(sample()+'\n'+sample().splitlines()[0],'glm',3)
 def test_decreasing_counter(self):
  with self.assertRaisesRegex(ValueError,'reset'):counters.compare(snap(sample(count=2)),snap(sample()))
 def test_created_reset_even_when_counts_increase(self):
  with self.assertRaisesRegex(ValueError,'timestamp'):counters.compare(snap(sample()),snap(sample(count=2,created=456)))
 def test_identity_change(self):
  b=snap(sample(count=2));b['identity']['pid']=2
  with self.assertRaisesRegex(ValueError,'identity'):counters.compare(snap(sample()),b)
 def test_label_change(self):
  with self.assertRaisesRegex(ValueError,'series'):counters.compare(snap(sample()),snap(sample(count=2).replace('worker="same"','worker="changed"')))
 def test_no_advance(self):
  with self.assertRaisesRegex(ValueError,'advance'):counters.compare(snap(sample()),snap(sample()))
 def test_inconsistent_accepted_positions(self):
  bad=sample(count=2).replace('position="1"} 12','position="1"} 13')
  with self.assertRaisesRegex(ValueError,'accounting'):counters.compare(snap(sample()),snap(bad))
 def test_invalid_counter(self):
  for value in ('nan','inf','-1','1.5'):
   with self.subTest(value=value),self.assertRaises(ValueError):counters.parse(sample().replace('} 10\n','} '+value+'\n'),'glm',3)

class Controls(unittest.TestCase):
 def test_actual_installed_pure_source_selectors(self):
  result=source_contract.validate();self.assertEqual(result["configuration_count"],16)
 def test_all_16_fixed_configs_and_shell_syntax(self):
  source=launch.BASE.read_text()
  for depth in (1,2,3,5):
   for slots in (1,2,4,8):
    c=launch.controls(depth,slots);sizes=c['compilation_config']['cudagraph_capture_sizes']
    self.assertEqual(sizes,sorted({(depth+1)*i for i in range(1,slots+1)} | (set(range(1,slots+1)) if depth>1 else set())))
    self.assertEqual(c['context_limit'],262144);self.assertEqual(c['memory_fraction'],.93)
    script=launch.variant(source,c)
    self.assertIn('-e VLLM_MXFP8_LM_HEAD=0 -e VLLM_MTP_NVFP4_LM_HEAD=0',script)
    self.assertIn('--kv-cache-dtype fp8_ds_mla --block-size 256',script)
    self.assertNotIn('--limit-mm-per-prompt',script)
    self.assertEqual(subprocess.run(['bash','-n'],input=script,text=True,capture_output=True).returncode,0)
 def test_invalid_controls(self):
  for d,s in ((0,1),(4,1),(True,1),(2,3),(2,16)):
   with self.assertRaises(ValueError):launch.controls(d,s)
 def test_unknown_source_rejected(self):
  with self.assertRaises(ValueError):launch.variant('echo changed',launch.controls(1,1))
 def test_sparse_draft_padding_is_valid(self):
  records=[record('ModelCudaGraphManager',range(1,9),4),record('SpeculatorCudaGraphManager',[1,2,4,8],4),record('SpeculatorCudaGraphManager',[4,8],1)]
  self.assertEqual(len(admission.captures(records,3,8)),3)
 def test_insufficient_draft_capacity(self):
  records=[record('ModelCudaGraphManager',range(1,9),4),record('SpeculatorCudaGraphManager',[1,2,4],4),record('SpeculatorCudaGraphManager',[8],1)]
  with self.assertRaisesRegex(ValueError,'draft-prefill'):admission.captures(records,3,8)
 def test_profile_sample_not_admission(self):
  with self.assertRaisesRegex(ValueError,'target'):admission.captures([record('ModelCudaGraphManager',[1],2,True)],1,1)
 def test_depth_one_does_not_require_extra_decode(self):
  admission.captures([record('ModelCudaGraphManager',[1],2),record('SpeculatorCudaGraphManager',[1],2)],1,1)
 def test_resources_are_actual_not_constant(self):
  log='Model loading took 97.89 GiB\nAvailable KV cache memory: 11.2 GiB\n4.78 GiB for peak activation\n0.13 GiB for CUDAGraph memory\nGPU KV cache size: 751,007 tokens\nSetting attention block size to 8,704 tokens'
  self.assertEqual(admission.resource_numbers(log)['kv_capacity_tokens'],751007)
  with self.assertRaises(ValueError):admission.resource_numbers(log.replace('751,007','131,072'))

class Semantic(unittest.TestCase):
 def test_natural_stop_and_exact_output_required(self):
  task={'validator':{'type':'json_equals','expected':{'answer':42}}}
  self.assertTrue(measure.validate_task({'success':True,'finish_reason':'stop','content':'{"answer":42}'},task))
  for content,finish in (('{"answer":43}','stop'),('{"answer":42}','length'),('```json\n{"answer":42}\n```','stop')):
   self.assertFalse(measure.validate_task({'success':True,'finish_reason':finish,'content':content},task))
 def test_prose_is_separate_exact_contract(self):
  self.assertTrue(measure.validate_task({'success':True,'finish_reason':'stop','content':'Hello.\n'},{'validator':{'type':'exact_text','expected':'Hello.'}}))

class Plugin(unittest.TestCase):
 def test_real_entrypoint_discovery_from_frozen_plan(self):
  plugin=HERE/"plans/d2-c1/plugin"
  code="from importlib.metadata import entry_points; p=[e for e in entry_points(group='vllm.general_plugins') if e.name=='glm_depth_receipts']; assert len(p)==1; assert callable(p[0].load()); print('ENTRYPOINT_DISCOVERED_WITHOUT_VLLM_OR_GPU_IMPORT')"
  env={**os.environ,"PYTHONPATH":str(plugin)}
  result=subprocess.run([sys.executable,"-c",code],env=env,capture_output=True,text=True)
  self.assertEqual(result.returncode,0,result.stderr)
 def test_capture_receipt_only_after_success_and_idempotent(self):
  class Manager:
   def __init__(self):
    self.graphs=[];self._max_full_descs_to_capture=None;self._capture_mem_samples=None
   def capture(self,fail=False):
    if fail:raise RuntimeError('capture failed')
    return 'unchanged result'
  mod=types.ModuleType('vllm.v1.worker.gpu.cudagraph_utils');mod.CudaGraphManager=Manager
  with tempfile.TemporaryDirectory() as temp,patch.dict(sys.modules,{'vllm.v1.worker.gpu.cudagraph_utils':mod}),patch.dict('os.environ',{'GLM53_CAPTURE_RECEIPTS':temp}):
   capture_plugin.install();wrapped=Manager.capture;capture_plugin.install();self.assertIs(wrapped,Manager.capture)
   with self.assertRaises(RuntimeError):Manager().capture(True)
   self.assertEqual(list(pathlib.Path(temp).glob('*.json')),[])
   self.assertEqual(Manager().capture(),'unchanged result');files=list(pathlib.Path(temp).glob('*.json'));self.assertEqual(len(files),1)
   self.assertFalse(json.loads(files[0].read_text())['profile_sample'])

if __name__=='__main__':unittest.main()
