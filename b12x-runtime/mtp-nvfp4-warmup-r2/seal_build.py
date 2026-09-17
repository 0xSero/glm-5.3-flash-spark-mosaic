from pathlib import Path
import subprocess,json,hashlib,shlex,datetime
p=Path(__file__).resolve().parent;h=['ssh','-i','~/.ssh/dgx-spark-node','valentine@spark-de5c.internal'];new='sha256:6c6581ea1454f013fa57f7c012532bf9a2c72a5a22ab55e20f51ca46df80bf8e';base='sha256:49e834851ff47ff01f3601122936f692cde75f253670aef0f895e44e40174ae0'
a=json.loads(subprocess.check_output(h+['docker image inspect '+base+' '+new]));b,n=a;assert n['RootFS']['Layers'][:len(b['RootFS']['Layers'])]==b['RootFS']['Layers']
for key in ('Env','Entrypoint','Cmd'):assert b['Config'].get(key)==n['Config'].get(key)
cpu=json.loads(subprocess.check_output(h+['docker inspect glm53-mtp-nvfp4-lifecycle-cpu-r2']))[0];assert cpu['Image']==new and cpu['State']['ExitCode']==0 and not cpu['State']['Running'] and not cpu['State']['OOMKilled']
code='import importlib.util,pathlib,hashlib,json; p=pathlib.Path(importlib.util.find_spec("vllm").origin).parent/"model_executor/warmup/b12x_warmup.py"; print(hashlib.sha256(p.read_bytes()).hexdigest())'
actual=subprocess.check_output(h+['docker run --rm --log-driver=none --entrypoint python3 '+new+' -c '+shlex.quote(code)],text=True).strip();assert actual==json.loads((p/'PINS.json').read_text())['after_sha256']
assert json.loads((p/'cpu.log').read_text().splitlines()[-1])['success']
(p/'image-inspects.json').write_text(json.dumps(a,indent=2)+'\n');(p/'cpu-inspect.json').write_text(json.dumps(cpu,indent=2)+'\n')
r={'state':'INSTALLED_CPU_LIFECYCLE_PASS_FULL_SERVING_PENDING','recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'base_image':base,'image':new,'warmup_sha256':actual,'new_image_bytes':n['Size']-b['Size'],'base_layers_env_entrypoint_cmd_preserved':True,'tests':5,'cuda_in_cpu_tests':False,'sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in p.iterdir() if f.is_file() and f.name!='BUILD_RECEIPT.json'}}
(p/'BUILD_RECEIPT.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='sha256'}))
