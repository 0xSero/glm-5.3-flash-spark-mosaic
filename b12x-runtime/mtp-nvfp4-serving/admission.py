"""Admit an already-started candidate using actual identity/capture/resource evidence."""
import argparse,datetime,json,pathlib,re,urllib.request
from common import docker,identity,sha,write

def option(argv,name):
 if argv.count(name)!=1:raise ValueError('Missing/duplicate runtime option '+name)
 return argv[argv.index(name)+1]

def resource_numbers(log):
 patterns={'model_memory_gib':r'Model loading took ([0-9.]+) GiB','kv_memory_gib':r'Available KV cache memory: ([0-9.]+) GiB','activation_gib':r'([0-9.]+) GiB for peak activation','graph_pool_actual_gib':r'([0-9.]+) GiB for CUDAGraph memory','kv_capacity_tokens':r'GPU KV cache size: ([0-9,]+) tokens','actual_hybrid_block_size':r'Setting attention block size to ([0-9,]+) tokens'}
 result={}
 for key,pattern in patterns.items():
  values=re.findall(pattern,log)
  if not values:raise ValueError('Missing actual resource-admission field '+key)
  result[key]=int(values[-1].replace(',','')) if key.endswith(('tokens','size')) else float(values[-1])
 if result['kv_capacity_tokens']<262144 or min(result[k] for k in ('model_memory_gib','kv_memory_gib','activation_gib'))<=0:raise ValueError('Context/resources not admitted')
 return result

def captures(records,depth,slots):
 actual=[r for r in records if r.get('capture_completed') is True and r.get('profile_sample') is False]
 def has(cls,n,q):
  return any(r['manager_class']==cls and any(d['cg_mode']=='FULL' and d['num_active_loras']==0 and d['num_reqs']==n and d['num_tokens']==n*q and d['uniform_token_count']==q and (d.get('max_query_len') is None or d['max_query_len']>=q) for d in r['descriptors']) for r in actual)
 if not all(has('ModelCudaGraphManager',n,depth+1) for n in range(1,slots+1)):raise ValueError('Actual target capture descriptors incomplete')
 if not all(any(has('SpeculatorCudaGraphManager',capacity,depth+1) for capacity in range(n,slots+1)) for n in range(1,slots+1)):raise ValueError('Actual draft-prefill capture coverage incomplete')
 if depth>1 and not all(any(has('SpeculatorCudaGraphManager',capacity,1) for capacity in range(n,slots+1)) for n in range(1,slots+1)):raise ValueError('Actual draft-decode capture coverage incomplete')
 return actual

def validate_inspect(info,plan):
 if info['Image']!=plan['image_id'] or not info['State']['Running'] or info['State']['OOMKilled']:raise ValueError('Runtime identity/state mismatch')
 argv=info['Config']['Cmd'];env=dict(x.split('=',1) for x in info['Config'].get('Env',[]) if '=' in x)
 expected={'--model':'/model','--served-model-name':'glm-5.3-flash','--host':'127.0.0.1','--port':'18080','--tensor-parallel-size':'1','--decode-context-parallel-size':'1','--quantization':'exl3','--dtype':'bfloat16','--kv-cache-dtype':'fp8_ds_mla','--block-size':'256','--gpu-memory-utilization':'0.93','--max-model-len':'262144','--max-num-seqs':str(plan['slots']),'--max-num-batched-tokens':'2048','--attention-backend':'B12X','--mm-processor-cache-gb':'0.1'}
 for name,value in expected.items():
  if option(argv,name)!=value:raise ValueError('Runtime option changed '+name)
 for name in ('--no-enable-expert-parallel','--enable-chunked-prefill','--no-enable-prefix-caching'):
  if name not in argv:raise ValueError('Required runtime option missing '+name)
 if any(x in argv for x in ('--enforce-eager','--language-model-only','--limit-mm-per-prompt')):raise ValueError('Graph or vision configuration changed')
 for name in ('speculative','compilation'):
  if json.loads(option(argv,'--'+name+'-config'))!=plan[name+'_config']:raise ValueError(name+' config mismatch')
 if json.loads(option(argv,'--additional-config'))!={'kda_prefill_backend':'b12x'}:raise ValueError('KDA backend changed')
 for name,value in {'VLLM_MXFP8_LM_HEAD':'0','VLLM_MTP_NVFP4_LM_HEAD':'0','GLM53_MTP_EXPERT_FP8':'0','GLM53_MTP_EXPERT_NVFP4':'1','VLLM_B12X_MOE_FP4_FORCE_A16':'1','VLLM_USE_V2_MODEL_RUNNER':'1'}.items():
  if env.get(name)!=value:raise ValueError('Protected precision/runner environment changed '+name)
 mounts={m['Destination']:m for m in info['Mounts']}
 for dest,key in (('/model','model_root'),('/mtp','mtp_root')):
  if mounts[dest]['RW'] or pathlib.Path(mounts[dest]['Source']).resolve()!=pathlib.Path(plan[key]).resolve():raise ValueError('Native model mount mismatch')
 return argv

def admit(run,endpoint):
 run=pathlib.Path(run);plan=json.loads((run/'plan.json').read_text())
 if plan['state'] not in ('STARTED_NOT_ADMITTED','ADMITTED'):raise ValueError('Candidate has not been launched')
 for name,digest in plan['metadata_sha256'].items():
  if sha(name)!=digest:raise ValueError('Source model/native metadata changed')
 ident=identity(plan['container_name']);info=json.loads(docker('inspect',ident['container_id']))[0];validate_inspect(info,plan)
 if sha(run/'launch.sh')!=plan['launcher_sha256'] or sha(run/'plugin/capture_plugin.py')!=plan['plugin_sha256']:raise ValueError('Frozen launcher/plugin changed')
 log=docker('logs',ident['container_id']).decode(errors='replace')
 records=[json.loads(p.read_text()) for p in sorted((run/'profiles/capture-receipts').glob('capture-*.json'))]
 actual=captures(records,plan['depth'],plan['slots']);resources=resource_numbers(log)
 req=urllib.request.Request(endpoint.rstrip('/')+'/v1/models')
 import os
 if os.environ.get('MODEL_API_KEY'):req.add_header('Authorization','Bearer '+os.environ['MODEL_API_KEY'])
 with urllib.request.urlopen(req,timeout=15) as response:models=json.load(response)
 if [m['id'] for m in models['data']]!=[plan['model']]:raise ValueError('Fresh served model identity mismatch')
 if identity(plan['container_name'])!=ident:raise ValueError('Runtime process changed during admission')
 evidence=run/'admission-evidence';evidence.mkdir(exist_ok=False)
 write(evidence/'inspect.json',info);(evidence/'engine.log').write_text(log);write(evidence/'models.json',models);write(evidence/'captures.json',actual);write(evidence/'identity.json',ident)
 receipt={'schema':'glm53-depth-runtime-admission-v1','state':'ADMITTED_FOR_ISOLATED_BENCHMARK_NOT_QUALITY_ACCEPTED','recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model':plan['model'],'depth':plan['depth'],'context_limit':262144,'max_num_seqs':plan['slots'],'max_num_batched_tokens':2048,'prefix_caching':False,'cuda_graphs':True,'gpu_memory_fraction':0.93,'kv_dtype':'fp8_ds_mla','speculative_config':plan['speculative_config'],'compilation_config':plan['compilation_config'],'identity':ident,**resources,'evidence_sha256':{p.name:sha(p) for p in evidence.iterdir()},'plan_sha256':sha(run/'plan.json')}
 write(run/'admission.json',receipt);return receipt

def verify(run):
 run=pathlib.Path(run);r=json.loads((run/'admission.json').read_text());plan=json.loads((run/'plan.json').read_text())
 for name,digest in plan['metadata_sha256'].items():
  if sha(name)!=digest:raise ValueError('Source model/native metadata changed')
 if r['plan_sha256']!=sha(run/'plan.json'):raise ValueError('Launch plan changed after admission')
 for name,digest in r['evidence_sha256'].items():
  if sha(run/'admission-evidence'/name)!=digest:raise ValueError('Admission evidence changed')
 if identity(plan['container_name'])!=r['identity']:raise ValueError('Admitted runtime changed or restarted')
 return r
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=pathlib.Path,required=True);p.add_argument('--endpoint',default='http://127.0.0.1:18080');a=p.parse_args();admit(a.run,a.endpoint)
