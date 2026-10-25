#!/usr/bin/env python3
import json,os,pathlib,re,subprocess,tempfile,time,urllib.request,sys
cli='/usr/bin/magnitude' if os.uname().sysname=='Linux' else str(pathlib.Path.home()/'.magnitude/bin/magnitude')
root=pathlib.Path.home()/'installer-audit-next'
profile=root/'runtime-profile';profile.mkdir(exist_ok=True)
(profile/'config.json').write_text('{"autoDownloadUpdates":false}')
env=dict(os.environ,MAGNITUDE_RELEASE_BASE_URL='http://127.0.0.1:18444',MAGNITUDE_DEV_DATA_DIR=str(profile),MAGNITUDE_DESKTOP_STATE_DIR=str(root/'runtime-state'),MAGNITUDE_DEV_PORT='11249')
env.pop('MAGNITUDE_ICN_PATH',None)
model='minicpm5-1b:gguf:q4'
def cmd(*args,timeout=60,fail=True):
 r=subprocess.run([cli,*args],env=env,capture_output=True,text=True,timeout=timeout)
 if fail and r.returncode: raise RuntimeError(' '.join(args)+': '+r.stderr+r.stdout)
 return r

def doc(*args):
 s=cmd(*args,'--json').stdout
 v=json.loads(s)
 print('JSON',json.dumps(args),'OK',s[:1500],flush=True)
 return v
release_log=(root/'release-http.log').open('w')
release_server=subprocess.Popen(['python3','-u','-m','http.server','18444','--bind','127.0.0.1','--directory',str(root/'release')],stdout=release_log,stderr=release_log)
for i in range(100):
 try:
  with urllib.request.urlopen('http://127.0.0.1:18444/@magnitudedev/cli@0.2.10/magnitude-release.json',timeout=1) as response: json.load(response)
  break
 except OSError: time.sleep(.1)
else: release_server.terminate();release_server.wait();raise RuntimeError('Release fixture server did not start')
with (root/'runtime-serve.log').open('w') as log:
 owner=subprocess.Popen([cli,'serve'],env=env,stdout=log,stderr=log)
 try:
  for i in range(120):
   if owner.poll() is not None: raise RuntimeError((root/'runtime-serve.log').read_text()[-8000:])
   if re.search(r'Runtime\s+Ready',cmd('status',fail=False).stdout): break
   time.sleep(1)
  else: raise RuntimeError('service did not become ready')
  print(cmd('status').stdout,flush=True)
  doc('hardware');doc('catalog','status');doc('catalog','list');doc('models','status');doc('hf','status')
  folder=root/'Empty folder with spaces';folder.mkdir(exist_ok=True)
  doc('models','folders','add',str(folder));locations=doc('models','folders','list')
  assert str(folder) in json.dumps(locations),locations
  doc('models','folders','remove',str(folder))
  assert str(folder) not in json.dumps(doc('models','folders','list'))
  # One real search checks newly shipped command-to-service compatibility; no login needed.
  search=doc('catalog','search','MiniCPM5','--limit','1')
  if '--inference' in sys.argv:
   refused=cmd('catalog','pull',model,'--drafter','none','--json',fail=False)
   assert refused.returncode != 0 and 'drafter' in (refused.stdout+refused.stderr),refused
   print('DRAFTER_REFUSAL',refused.stdout+refused.stderr,flush=True)
   print('PULL',cmd('catalog','pull',model,'--json').stdout,flush=True)
   deadline=time.monotonic()+300
   while time.monotonic()<deadline:
    text=cmd('models','status',model).stdout
    if re.search(r'Installation\s+Installed',text): break
    if re.search(r'(Installation|Runtime)\s+Failed',text): raise RuntimeError(text)
    time.sleep(2)
   else: raise RuntimeError('Model acquisition timeout: '+text)
   print('LOAD',cmd('models','load',model,'--json').stdout,flush=True)
   deadline=time.monotonic()+240
   while time.monotonic()<deadline:
    text=cmd('models','status',model).stdout
    if re.search(r'Runtime\s+(Ready|Loaded)',text): break
    if re.search(r'Runtime\s+Failed',text): raise RuntimeError(text)
    time.sleep(2)
   else: raise RuntimeError('Model load timeout: '+text)
   print('MODEL',text,flush=True)
   doc('models','status',model)
   data=json.dumps({'model':model,'messages':[{'role':'user','content':'Reply with OK.'}],'max_tokens':16,'temperature':0,'stream':False}).encode()
   req=urllib.request.Request('http://127.0.0.1:11249/inference/v1/chat/completions',data=data,headers={'Content-Type':'application/json'})
   with urllib.request.urlopen(req,timeout=120) as response: result=json.load(response)
   assert result.get('usage',{}).get('completion_tokens',0)>0 and result.get('choices'),result
   print('COMPLETION',json.dumps(result),flush=True)
   doc('models','stop',model)
  print('RESULT=next-source-runtime-new-cli-checks-passed',flush=True)
 finally:
  owner.terminate()
  try: owner.wait(timeout=30)
  except subprocess.TimeoutExpired: owner.kill();owner.wait()
  release_server.terminate();release_server.wait();release_log.close()
  requests=(root/'release-http.log').read_text()
  print('RELEASE_ACQUISITION_REQUESTS',requests[-2500:],flush=True)
  assert 'magnitude-icn-base-' in requests, 'Engine artifact was not requested from the release fixture'
