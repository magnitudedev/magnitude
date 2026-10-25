#!/usr/bin/env python3
import json,os,pathlib,re,subprocess,time,urllib.request
root=pathlib.Path.home()/'installer-audit-next';cli='/usr/bin/magnitude' if os.uname().sysname=='Linux' else str(pathlib.Path.home()/'.magnitude/bin/magnitude')
env=dict(os.environ,MAGNITUDE_DEV_DATA_DIR=str(root/'runtime-profile'),MAGNITUDE_DESKTOP_STATE_DIR=str(root/'runtime-state'),MAGNITUDE_DEV_PORT='11249',MAGNITUDE_RELEASE_BASE_URL='http://127.0.0.1:9');env.pop('MAGNITUDE_ICN_PATH',None)
model='minicpm5-1b:gguf:q4'
def cmd(*args):
 r=subprocess.run([cli,*args],env=env,capture_output=True,text=True,timeout=30)
 assert r.returncode==0,(args,r.stdout,r.stderr)
 return r.stdout
with (root/'api-serve.log').open('w') as log:
 owner=subprocess.Popen([cli,'serve'],env=env,stdout=log,stderr=log)
 try:
  for i in range(90):
   if owner.poll() is not None:raise RuntimeError((root/'api-serve.log').read_text()[-5000:])
   if re.search(r'Runtime\s+Ready',cmd('status')):break
   time.sleep(1)
  else:raise RuntimeError('Not ready')
  print('LOAD',cmd('models','load',model,'--json'),flush=True)
  for i in range(120):
   state=cmd('models','status',model)
   if re.search(r'Runtime\s+Ready',state):break
   if re.search(r'Runtime\s+Failed',state):raise RuntimeError(state)
   time.sleep(1)
  else:raise RuntimeError('Not loaded')
  for streaming in [False,True]:
   body={'model':model,'messages':[{'role':'user','content':'Reply with the single word OK.'}],'max_tokens':512,'temperature':0,'reasoning_effort':'none','stream':streaming}
   request=urllib.request.Request('http://127.0.0.1:11249/inference/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
   with urllib.request.urlopen(request,timeout=120) as response:raw=response.read().decode()
   if not streaming:
    value=json.loads(raw);content=value['choices'][0]['message'].get('content') or ''
    assert content.strip() and 'OK' in content,value
    print('FINAL_COMPLETION',json.dumps(value),flush=True)
   else:
    assert 'data: [DONE]' in raw,raw
    chunks=[json.loads(line[6:]) for line in raw.splitlines() if line.startswith('data: {')]
    content=''.join(choice['delta'].get('content') or '' for chunk in chunks for choice in chunk.get('choices',[]))
    assert 'OK' in content,raw
    print('STREAM_COMPLETION',json.dumps({'content':content,'chunks':len(chunks),'done':True}),flush=True)
  print('STOP',cmd('models','stop',model,'--json'),flush=True)
  print('RESULT=next-complete-answer-and-streaming-passed',flush=True)
 finally:
  owner.terminate()
  try:owner.wait(timeout=30)
  except subprocess.TimeoutExpired:owner.kill();owner.wait()
