"""Freeze a depth/slot variant of the existing guarded launcher; opt-in execution."""
import argparse,hashlib,json,os,pathlib,shutil,subprocess
from common import ROOT,docker,locked_dependencies,sha,write
from source_contract import validate as validate_source
BASE=ROOT/'b12x-runtime/start-b12x-candidate.sh'
SOURCE_SNAPSHOT=ROOT/'b12x-runtime/sm121-router-audit/mtp-depth-installed-sources.json'

def controls(depth,slots):
 if type(depth) is not int or depth not in (1,2,3,5) or type(slots) is not int or slots not in (1,2,4,8):raise ValueError('Depth1/2/3/5 and slots1/2/4/8 only')
 sizes=sorted({(depth+1)*n for n in range(1,slots+1)} | (set(range(1,slots+1)) if depth>1 else set()))
 return {'depth':depth,'slots':slots,'context_limit':262144,'chunk':2048,'memory_fraction':0.93,'speculative_config':{'method':'mtp','model':'/mtp','num_speculative_tokens':depth,'attention_backend':'B12X'},'compilation_config':{'cudagraph_mode':'FULL_DECODE_ONLY','custom_ops':['all'],'cudagraph_capture_sizes':sizes,'max_cudagraph_capture_size':sizes[-1]}}

def variant(source,control):
 changes=[('--max-num-seqs 1 --max-num-batched-tokens 2048',f"--max-num-seqs {control['slots']} --max-num-batched-tokens 2048"),
 ("--compilation-config '{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"custom_ops\":[\"all\"],\"cudagraph_capture_sizes\":[2],\"max_cudagraph_capture_size\":2}'","--compilation-config '"+json.dumps(control['compilation_config'],separators=(',',':'))+"'"),
 ("--speculative-config '{\"method\":\"mtp\",\"model\":\"/mtp\",\"num_speculative_tokens\":1,\"attention_backend\":\"B12X\"}'","--speculative-config '"+json.dumps(control['speculative_config'],separators=(',',':'))+"'"),
 ('  "$GLM53_IMAGE" --model /model', '  -v "$GLM53_DEPTH_PLUGIN_ROOT":/depth_sweep:ro \\\n  -e PYTHONPATH=/depth_sweep -e GLM53_CAPTURE_RECEIPTS=/profiles/capture-receipts \\\n  "$GLM53_IMAGE" --model /model')]
 for old,new in changes:
  if source.count(old)!=1:raise ValueError('Guarded launcher source changed: '+old)
  source=source.replace(old,new)
 return source

def check_image(image):
 if not __import__('re').fullmatch(r'(?:[a-zA-Z0-9./:_-]+@)?sha256:[a-f0-9]{64}',image):raise ValueError('Immutable image digest or local image ID required')
 data=json.loads(docker('image','inspect',image))[0];env=dict(x.split('=',1) for x in data['Config'].get('Env',[]) if '=' in x)
 if env.get('PYTHONPATH') or env.get('VLLM_PLUGINS') is not None:raise ValueError('Image has an existing Python/plugin filter; preserve and review before adapting')
 expected={k:hashlib.sha256(v.encode()).hexdigest() for k,v in json.loads(SOURCE_SNAPSHOT.read_text()).items()}
 code='import hashlib,json,pathlib; b=pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm");print(json.dumps({k:hashlib.sha256((b/k).read_bytes()).hexdigest() for k in '+repr(list(expected))+'}))'
 actual=json.loads(docker('run','--rm','--network','none','--cpus','1','--memory','1g','--memory-swap','1g','--entrypoint','python3',image,'-c',code))
 if actual!=expected:raise ValueError('Installed depth/capture source differs from audited image')
 return data,actual

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--depth',type=int,required=True);p.add_argument('--slots',type=int,required=True);p.add_argument('--output',type=pathlib.Path,required=True)
 for name in ('image','model-root','mtp-root','container'):p.add_argument('--'+name)
 p.add_argument('--execute',action='store_true');a=p.parse_args();lock=locked_dependencies();c=controls(a.depth,a.slots);source_validation=validate_source()
 a.output.mkdir(parents=True,exist_ok=False);base=BASE.read_text();(a.output/'base-launcher.sh').write_text(base);candidate=a.output/'launch.sh';candidate.write_text(variant(base,c));subprocess.run(['bash','-n',str(candidate)],check=True)
 plugin=a.output/'plugin';plugin.mkdir();shutil.copy2(pathlib.Path(__file__).with_name('capture_plugin.py'),plugin/'capture_plugin.py')
 metadata=plugin/'glm_depth_receipts-0.0.0.dist-info';metadata.mkdir();(metadata/'METADATA').write_text('Metadata-Version: 2.1\nName: glm-depth-receipts\nVersion: 0.0.0\n');(metadata/'entry_points.txt').write_text('[vllm.general_plugins]\nglm_depth_receipts = capture_plugin:install\n')
 c.update(state='PLANNED_NOT_LAUNCHED',model='glm-5.3-flash',source_sha256=lock,launcher_sha256=sha(candidate),plugin_sha256=sha(plugin/'capture_plugin.py'))
 write(a.output/'plan.json',c);write(a.output/'source-contract.json',source_validation)
 if not a.execute:return
 if not all((a.image,a.model_root,a.mtp_root,a.container)):p.error('Execution requires image/model-root/mtp-root/container')
 info,source=check_image(a.image);write(a.output/'image-inspect.json',info);write(a.output/'installed-source-check.json',source)
 model=pathlib.Path(a.model_root).resolve();mtp=pathlib.Path(a.mtp_root).resolve()
 if not model.is_dir() or not mtp.is_dir():raise ValueError('Existing verified model and native MTP directories required')
 env=os.environ.copy();env.update(GLM53_IMAGE=a.image,GLM53_MODEL_ROOT=str(model),GLM53_MTP_ROOT=str(mtp),GLM53_CONTAINER_NAME=a.container,GLM53_PROFILE_ROOT=str((a.output/'profiles').resolve()),GLM53_DEPTH_PLUGIN_ROOT=str(plugin.resolve()))
 metadata_names=('config.json','source-config.json','quantization_config.json','model.safetensors.index.json','EXL3_MANIFEST.json','BUILD_STATUS.json','native-mtp-view.json')
 c.update(metadata_sha256={str(root/name):sha(root/name) for root in (model,mtp) for name in metadata_names if (root/name).is_file()},container_name=a.container,image_id=info['Id'],model_root=str(model),mtp_root=str(mtp),state='LAUNCH_ATTEMPTED_NOT_ADMITTED');write(a.output/'plan.json',c)
 with (a.output/'launch.log').open('w') as f:subprocess.run(['bash',str(candidate)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 c['state']='STARTED_NOT_ADMITTED';write(a.output/'plan.json',c)
if __name__=='__main__':main()
