"""Isolated unpruned Projection14 plus NVFP4 native draft capacity test."""
import datetime,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
from common import sha,write
ROOT=Path('/home/valentine/glm53-single-spark-release-20260911')
W=Path(__file__).resolve().parent
IMAGE='sha256:6c6581ea1454f013fa57f7c012532bf9a2c72a5a22ab55e20f51ca46df80bf8e'
def docker(*a):return subprocess.check_output(['docker',*a],text=True)
for n,h in json.loads((W/'SOURCE_FREEZE.json').read_text()).items():assert sha(W/n)==h,n
image=json.loads(docker('image','inspect',IMAGE))[0]
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU is owned'
assert shutil.disk_usage(ROOT).free>20*2**30
mem=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
assert mem>110*2**30,mem
cpu=json.loads(docker('inspect','glm53-mtp-nvfp4-lifecycle-cpu-r2'))[0]
assert cpu['Image']==IMAGE and not cpu['State']['Running'] and cpu['State']['ExitCode']==0 and not cpu['State']['OOMKilled']
assert '"success": true' in (ROOT/'mtp-nvfp4-warmup-r2/cpu.log').read_text()
proof=ROOT/'b12x-runtime/mtp-full-sparse-proof/attempt1'
r=json.loads((proof/'TERMINAL_RECEIPT.json').read_text())
for n,h in r['sha256'].items():assert sha(proof/n)==h,n
assert json.loads((proof/'full-sparse.json').read_text())['state']=='FULL288_SPARSE_MOE_REFERENCE_GRAPH_PASS_SERVING_PENDING'
model=ROOT/'quality/unpruned-next-candidate/projection14/model'
mtp=ROOT/'b12x-runtime/mtp-draft-nvfp4/native-mtp-view'
assert sha(model/'EXL3_MANIFEST.json')=='80967e71922a534a3ab2872ad90edc4aacaf958bcb0a060699e599318727556d'
assert sha(model/'config.json')=='a8f05d1b6fbbb14adb3cc358e22435253e6678910cf9aef49e7bd2716472292f'
assert sha(mtp/'model.safetensors.index.json')=='3b83745278ec1f43d45f4c53e141a18bce30e3ece70d969a76088b2d2c2f534d'
run=W/'attempt1';run.mkdir(exist_ok=False)
for n in ('launch.sh','admission.py','common.py','watch.py'):shutil.copy2(W/n,run/n)
plugin=run/'plugin';plugin.mkdir();shutil.copy2(W/'capture_plugin.py',plugin/'capture_plugin.py')
meta=plugin/'glm_depth_receipts-0.0.0.dist-info';meta.mkdir();(meta/'METADATA').write_text('Metadata-Version: 2.1\nName: glm-depth-receipts\nVersion: 0.0.0\n');(meta/'entry_points.txt').write_text('[vllm.general_plugins]\nglm_depth_receipts = capture_plugin:install\n')
plan={'state':'LAUNCH_ATTEMPTED_NOT_ADMITTED','model':'glm-5.3-flash','container_name':'glm53-projection14-nvfp4mtp-attempt1','image_id':IMAGE,'model_root':str(model),'mtp_root':str(mtp),'depth':1,'slots':1,
'context_limit':262144,'chunk':2048,'memory_fraction':.93,'metadata_sha256':{},'launcher_sha256':sha(run/'launch.sh'),'plugin_sha256':sha(plugin/'capture_plugin.py'),
'speculative_config':{'method':'mtp','model':'/mtp','num_speculative_tokens':1,'attention_backend':'B12X'},'compilation_config':{'cudagraph_mode':'FULL_DECODE_ONLY','custom_ops':['all'],'cudagraph_capture_sizes':[2],'max_cudagraph_capture_size':2},'MemAvailable_before':mem,'scope':'Unpruned Projection14 full-server capacity and function; quality not accepted; not a matched draft speed A/B.'}
for path in (model,mtp):
 for n in ('config.json','source-config.json','model.safetensors.index.json','EXL3_MANIFEST.json','native-mtp-view.json'):
  if (path/n).is_file():plan['metadata_sha256'][str(path/n)]=sha(path/n)
write(run/'plan.json',plan);write(run/'image-inspect.json',image);write(run/'cpu-inspect.json',cpu)
env=os.environ.copy();env.update(GLM53_IMAGE=IMAGE,GLM53_MODEL_ROOT=str(model),GLM53_MTP_ROOT=str(mtp),GLM53_CONTAINER_NAME=plan['container_name'],GLM53_PROFILE_ROOT=str(run/'profiles'),GLM53_DEPTH_PLUGIN_ROOT=str(plugin))
with (run/'launch.log').open('x') as f:subprocess.run(['bash',str(run/'launch.sh')],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
plan['state']='STARTED_NOT_ADMITTED';write(run/'plan.json',plan)
with (run/'watch.log').open('x') as f:p=subprocess.Popen([sys.executable,str(run/'watch.py')],cwd=run,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
write(run/'watch-process.json',{'pid':p.pid,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
print(json.dumps({'container':plan['container_name'],'watch_pid':p.pid}))
