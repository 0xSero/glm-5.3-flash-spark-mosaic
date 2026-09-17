import datetime,hashlib,json,pathlib,shutil,subprocess
ids=['c6c0c2d71956','a1db32da841c','aaaae4784444','fdabb37cbf14','cf8c965ccf88','56754c181d7a','7d3d3b59e78b','18f3f3207753']
root=pathlib.Path('/home/sero/work/glm53-single-spark-release-20260911/b12x-runtime/retired-container-cleanup-20260912')
root.mkdir(parents=True,exist_ok=False)
def cmd(*args): return subprocess.check_output(args)
image='sha256:4f16e6cab3ca3966a325cb8351b1ea58a194e471a23197019e12e891ae17a959'
(root/'retained-image-inspect.json').write_bytes(cmd('docker','image','inspect',image))
(root/'retained-wheels.txt').write_bytes(cmd('docker','run','--rm','--network','none','--cpus','1','--memory','1g','--memory-swap','1g','--entrypoint','sh',image,'-c','ls -l /opt/wheels'))
rows=[]
for cid in ids:
 data=cmd('docker','inspect','--size',cid); item=json.loads(data)[0]
 assert item['Id'].startswith(cid) and item['State']['Status']=='exited' and not item['State']['Running'],item
 assert 'compile' in item['Name'] or 'build' in item['Name'],item['Name']
 (root/(cid+'-inspect.json')).write_bytes(data)
 with (root/(cid+'-logs.txt')).open('wb') as f: subprocess.run(['docker','logs',cid],stdout=f,stderr=subprocess.STDOUT,check=True)
 rows.append({'id':item['Id'],'name':item['Name'],'size_rw':item.get('SizeRw'),'exit_code':item['State']['ExitCode']})
report={'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'before_free_bytes':shutil.disk_usage('/').free,'retained_image':image,'containers':rows,'removed':[]}
(root/'preflight.json').write_text(json.dumps(report,indent=2))
for row in rows:
 fresh=json.loads(cmd('docker','inspect',row['id']))[0]
 assert fresh['State']['Status']=='exited' and not fresh['State']['Running']
 cmd('docker','rm',row['id'])
 report['removed'].append(row['id'])
 (root/'result.json').write_text(json.dumps(report,indent=2))
report['after_free_bytes']=shutil.disk_usage('/').free
report['reclaimed_free_bytes']=report['after_free_bytes']-report['before_free_bytes']
report['state']='EIGHT_ALLOWLISTED_TERMINAL_CONTAINERS_REMOVED_IMAGES_PRESERVED'
report['evidence_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.name!='result.json'}
(root/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
