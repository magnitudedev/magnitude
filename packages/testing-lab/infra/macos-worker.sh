#!/bin/bash
set -euo pipefail
umask 077
# Administrator-owned preparation runs before candidate source or run credentials arrive.
exec /opt/homebrew/bin/python3 - "$1" <<'PY'
import hashlib,json,os,pathlib,platform,pwd,re,shlex,shutil,subprocess,sys,tomllib,urllib.request
config_path=pathlib.Path(sys.argv[1])
config=json.loads(config_path.read_text())
config_path.unlink()
def output(*args):return subprocess.check_output(args,text=True).strip()
if platform.system()!='Darwin' or platform.machine()!='arm64':raise SystemExit('Mac preparation requires native Apple Silicon')
account=pwd.getpwuid(os.getuid())
if account.pw_name!='runner' or account.pw_dir!='/Users/runner' or account.pw_uid==0:raise SystemExit('Unexpected Namespace guest user')
if output('stat','-f','%Su','/dev/console')!=account.pw_name:raise SystemExit('Namespace guest has no admitted interactive desktop')
if output('sw_vers','-productVersion')!=config['productVersion'] or output('sw_vers','-buildVersion')!=config['buildVersion']:raise SystemExit('Mac preparation image mismatch')
home=pathlib.Path(account.pw_dir)
root=home/'lab-runtime'
state=pathlib.Path('/var/db/magnitude-lab')
receipt=state/'runtime.json'
if receipt.exists():
 if receipt.stat().st_uid!=0 or receipt.stat().st_mode & 0o022:raise SystemExit('Unsafe Mac runtime receipt')
 previous=json.loads(receipt.read_text())
 if previous['identity']!=config['identity']:raise SystemExit('Mac runtime belongs to a different preparation recipe')
 if not (root/'worker').is_file():raise SystemExit('Prepared Mac runtime launcher is missing')
 print('Mac runtime already prepared')
 raise SystemExit(0)
if root.exists():raise SystemExit('Incomplete Mac preparation requires a fresh worker')
if shutil.disk_usage(home).free < 40*1024**3:raise SystemExit('Mac worker has less than 40 GiB free')
subprocess.run(['sudo','-n','true'],check=True)
for command in [['xcodebuild','-version'],['xcrun','--find','clang'],['cmake','--version'],['git','--version']]:subprocess.run(command,check=True)
root.mkdir(mode=0o700)
class HttpsRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,req,fp,code,msg,headers,newurl):
  if not newurl.startswith('https://'):raise SystemExit('Insecure native-tool redirect')
  return super().redirect_request(req,fp,code,msg,headers,newurl)
class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,*args,**kwargs):return None
for key in ['runtime','node','rustup','tirith']:
 item=config[key]; destination=root/('download-'+key); digest=hashlib.sha256(); total=0
 try:
  client=urllib.request.build_opener(HttpsRedirect if key=='tirith' else NoRedirect)
  with client.open(item['url'],timeout=120) as response,destination.open('xb') as out:
   for chunk in iter(lambda:response.read(1024*1024),b''):
    total+=len(chunk)
    if total>item['bytes']:raise ValueError('excess bytes')
    digest.update(chunk);out.write(chunk)
 except Exception:raise SystemExit('Pinned '+key+' download failed') from None
 if total!=item['bytes'] or digest.hexdigest()!=item['sha256']:raise SystemExit('Pinned '+key+' integrity mismatch')
(root/'node-bin').mkdir()
subprocess.run(['tar','-xzf',str(root/'download-node'),'--strip-components=1','-C',str(root/'node-bin')],check=True)
subprocess.run(['tar','-xzf',str(root/'download-runtime'),'-C',str(root)],check=True)
workspace=root/'runtime'
hermes=json.loads((workspace/'packages/testing-lab/tools/hermes.json').read_text())
if hermes['repository']!='https://github.com/NousResearch/hermes-agent.git' or not re.fullmatch('[a-f0-9]{40}',hermes['commit']):raise SystemExit('Invalid Hermes source pin')
rust=tomllib.loads((workspace/'inference/rust-toolchain.toml').read_text())['toolchain']['channel']
if not re.fullmatch(r'\d+\.\d+\.\d+',rust):raise SystemExit('Rust channel must be pinned')
path=f'{home}/.local/bin:{root}/node-bin/bin:{root}/tooling/node_modules/.bin:{home}/.cargo/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin'
environment={**os.environ,'PATH':path,'CARGO_HOME':str(home/'.cargo'),'RUSTUP_HOME':str(home/'.rustup'),
 'LAB_RUST_VERSION':rust,'LAB_BUN_VERSION':config['bunVersion'],'LAB_HERMES_COMMIT':hermes['commit'],'LAB_HERMES_VERSION':hermes['version']}
(root/'download-rustup').rename(root/'rustup-init');(root/'rustup-init').chmod(0o700)
try:
 subprocess.run(['/bin/bash','-c',r'''
set -euo pipefail
../rustup-init -y --profile minimal --default-toolchain "$LAB_RUST_VERSION" --no-modify-path
npm install --prefix ../tooling --no-audit --no-fund "bun@$LAB_BUN_VERSION"
test "$(bun --version)" = "$LAB_BUN_VERSION"
bun install --frozen-lockfile --ignore-scripts
bun packages/version/scripts/generate-version.ts
npm ci --prefix packages/testing-lab/tools --no-audit --no-fund
/opt/homebrew/bin/python3 -m venv ../python-tools
../python-tools/bin/pip install --disable-pip-version-check --no-deps --require-hashes -r packages/testing-lab/tools/python-tools.txt
git init -q ../hermes-agent
git -C ../hermes-agent remote add origin https://github.com/NousResearch/hermes-agent.git
git -C ../hermes-agent fetch -q --depth=1 origin "$LAB_HERMES_COMMIT"
git -C ../hermes-agent checkout -q --detach FETCH_HEAD
test "$(git -C ../hermes-agent rev-parse HEAD)" = "$LAB_HERMES_COMMIT"
../python-tools/bin/uv python install 3.12.13
../python-tools/bin/uv sync --project ../hermes-agent --python 3.12.13 --frozen --no-dev
mkdir ../native-tools
tar -xzf ../download-tirith -C ../native-tools tirith
mkdir -p "$HOME/.local/bin"
ln -s "$PWD/packages/testing-lab/tools/node_modules/.bin/pi" "$HOME/.local/bin/pi"
ln -s "$PWD/packages/testing-lab/tools/node_modules/.bin/opencode" "$HOME/.local/bin/opencode"
ln -s "$PWD/../hermes-agent/.venv/bin/hermes" "$HOME/.local/bin/hermes"
ln -s "$PWD/../native-tools/tirith" "$HOME/.local/bin/tirith"
pi --version
opencode --version
hermes --version
tirith --version
../hermes-agent/.venv/bin/python -c 'import importlib.metadata,os; assert importlib.metadata.version("hermes-agent")==os.environ["LAB_HERMES_VERSION"]'
(cd packages/testing-lab && node -e 'const pty=require("node-pty"); const p=pty.spawn("/bin/sh",["-c","exit 0"],{});p.onExit(e=>process.exit(e.exitCode));setTimeout(()=>{p.kill();process.exit(1)},10000)')
bun -e 'await import("./packages/testing-lab/src/worker-entry.ts")'
'''],cwd=workspace,env=environment,check=True)
except subprocess.CalledProcessError as error:raise SystemExit('Mac dependency preparation exited '+str(error.returncode)) from None
launcher='\n'.join(['#!/bin/bash','set -euo pipefail','umask 022',
 'export PATH='+shlex.quote(path),'export CARGO_HOME='+shlex.quote(str(home/'.cargo')),
 'export RUSTUP_HOME='+shlex.quote(str(home/'.rustup')),
 'export LAB_TERMINAL_NODE_EXECUTABLE='+shlex.quote(str(root/'node-bin/bin/node')),
 'export LAB_PI_EXECUTABLE='+shlex.quote(str(home/'.local/bin/pi')),
 'export LAB_OPENCODE_EXECUTABLE='+shlex.quote(str(home/'.local/bin/opencode')),
 'export LAB_HERMES_EXECUTABLE='+shlex.quote(str(home/'.local/bin/hermes')),
 'cd '+shlex.quote(str(workspace)), 'exec bun packages/testing-lab/src/worker-entry.ts "$@"',''])
(root/'worker').write_text(launcher);(root/'worker').chmod(0o755)
local_receipt=root/'receipt.json'
local_receipt.write_text(json.dumps({'identity':config['identity'],'runtimeSha256':config['runtime']['sha256'],
 'uid':account.pw_uid,'productVersion':config['productVersion'],'buildVersion':config['buildVersion'],'rustVersion':rust}))
subprocess.run(['sudo','-n','install','-d','-o','root','-g','wheel','-m','755',str(state)],check=True)
subprocess.run(['sudo','-n','install','-o','root','-g','wheel','-m','644',str(local_receipt),str(receipt)],check=True)
local_receipt.unlink()
for key in ['runtime','node','tirith']:(root/('download-'+key)).unlink()
(root/'rustup-init').unlink()
print('Mac runtime, native terminal and interactive user prepared')
PY
