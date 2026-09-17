#!/usr/bin/env python3
"""Run on2822: stream immutable candidate557f->de5c, then queue reserved quality."""
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import time

ROOT=Path('/home/sero/work/glm53-single-spark-release-20260911/staging-k2-k256')
SOURCE='/home/valentine/glm53-single-spark-release-20260911/k2-massmax-k256'
DEST='/home/valentine/glm53-single-spark-release-20260911'
SRC_HOST='valentine@spark-raila.internal'
DST_HOST='valentine@spark-raila.internal'
PIN='7c7a2e47c7cde8996ac54c6f2ca5c3f7532b682d5d7296408c97a14e8950f73c'
IMAGE='sha256:54acf16064726697ab079dfbd297c1c1c260cb28867f8287f5ae2ec3bbb0a382'


def remote(host,args):return ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,shlex.join(args)]
def call(host,args):return subprocess.check_output(remote(host,args),text=True)
def status(state,**kw):
    value={'state':state,'time':time.time(),**kw}
    (ROOT/'status.json.tmp').write_text(json.dumps(value,indent=2)+'\n')
    (ROOT/'status.json.tmp').replace(ROOT/'status.json')
    print(json.dumps(value),flush=True)


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'stage.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source_check='''import pathlib,json,hashlib
r=pathlib.Path(%r);m=r/'EXL3_MANIFEST.json';s=json.loads((r/'BUILD_STATUS.json').read_text());assert s['state']=='COMPLETE';assert hashlib.sha256(m.read_bytes()).hexdigest()==s['manifest_sha256']==%r;print(json.dumps({'manifest_sha256':s['manifest_sha256'],'bytes':sum(p.stat().st_size for p in r.rglob('*.safetensors'))}))'''%(SOURCE,PIN)
    proof=json.loads(call(SRC_HOST,['python3','-c',source_check]))
    (ROOT/'source-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    setup='''from pathlib import Path
import shutil
r=Path(%r);p=r/'k2-massmax-k256';q=r/'staging-k2-k256';assert not p.exists(), 'candidate destination already exists; inspect before resuming';assert shutil.disk_usage(r).free>%d;p.mkdir();q.mkdir(exist_ok=True)'''%(DEST,proof['bytes']+60*2**30)
    call(DST_HOST,['python3','-c',setup])
    # Ship only our small scripts to the new queue directory.
    for name in ('queue_quality.py','validate_stage.py','seal_baseline.py'):
        data=(ROOT/name).read_bytes()
        command=remote(DST_HOST,['python3','-c',f"import sys;from pathlib import Path;Path({(DEST+'/staging-k2-k256/'+name)!r}).write_bytes(sys.stdin.buffer.read())"])
        subprocess.run(command,input=data,check=True)
    status('STREAMING',source_manifest_sha256=PIN,source_bytes=proof['bytes'])
    source=subprocess.Popen(remote(SRC_HOST,['tar','-C',SOURCE,'-cf','-','.']),stdout=subprocess.PIPE)
    destination=subprocess.Popen(remote(DST_HOST,['tar','-C',DEST+'/k2-massmax-k256','-xf','-']),stdin=source.stdout)
    source.stdout.close()
    target_rc=destination.wait();source_rc=source.wait()
    if target_rc or source_rc:raise RuntimeError(f'tar source={source_rc} destination={target_rc}; preserve partial destination')
    assert json.loads(call(SRC_HOST,['python3','-c',source_check]))==proof
    status('COPIED_VALIDATING_HASHES',source_manifest_sha256=PIN)
    cmd=['docker','run','--rm','--network','none','--memory','2g','--cpus','1',
         '-v',DEST+':/release:ro','-v',DEST+'/staging-k2-k256:/queue',
         '--entrypoint','python3',IMAGE,'/queue/validate_stage.py']
    validation=call(DST_HOST,cmd)
    (ROOT/'validation.stdout.txt').write_text(validation)
    status('STAGED_HASH_VERIFIED',source_manifest_sha256=PIN,validation=json.loads(validation))
    # Detached queue only watches the existing control until successful sealing.
    code='''import subprocess,pathlib,json
r=pathlib.Path(%r);log=(r/'queue.log').open('ab');p=subprocess.Popen(['python3','-u',str(r/'queue_quality.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);print(json.dumps({'queue_pid':p.pid}))'''%(DEST+'/staging-k2-k256')
    queued=json.loads(call(DST_HOST,['python3','-c',code]))
    status('QUALITY_QUEUED_AFTER_Q3_CONTROL',**queued,source_manifest_sha256=PIN)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        status('FAILED_PRESERVED',error=repr(exc))
        raise
