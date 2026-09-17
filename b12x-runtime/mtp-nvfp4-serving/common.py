import hashlib,json,pathlib,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[2]
def sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def write(path,data):
 path=pathlib.Path(path);temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n');temp.replace(path)
def docker(*args):return subprocess.check_output(['docker',*args],timeout=60,stderr=subprocess.STDOUT)
def identity(container):
 data=json.loads(docker('inspect',container))[0]
 if not data['State']['Running']:raise ValueError('Runtime container is not running')
 return {'container_id':data['Id'],'image_id':data['Image'],'started_at':data['State']['StartedAt'],'processes':[line.split() for line in docker('top',container,'-eo','pid,lstart,comm').decode().splitlines()[1:]]}
def locked_dependencies():
 lock=json.loads(pathlib.Path(__file__).with_name('dependencies.json').read_text())
 for name,digest in lock.items():
  if sha(ROOT/name)!=digest:raise ValueError('Dependency changed; review and refreeze: '+name)
 return lock
