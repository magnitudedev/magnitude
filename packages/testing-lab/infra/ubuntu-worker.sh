#!/bin/bash
set -euo pipefail
umask 077
# This file is installed by cloud-init before any run credential or candidate source arrives.
. /etc/os-release
test "$ID" = ubuntu && test "$VERSION_ID" = 24.04
export DEBIAN_FRONTEND=noninteractive
# Restarting the Azure agent while its readiness command runs can sever observation.
# Fresh application processes use the newly installed libraries after preparation.
export NEEDRESTART_MODE=l
apt-get update -qq
apt-get install -y -qq curl ca-certificates python3 python3-venv git xz-utils tar build-essential cmake libclang-dev libssl-dev pkg-config fakeroot rpm binutils nftables polkitd pkexec xvfb xauth dbus-x11 openbox libgtk-3-0t64 libnss3 libasound2t64 libgbm1 libxss1 libxtst6

python3 - <<'PY'
import hashlib,json,os,pathlib,platform,pwd,re,shlex,shutil,subprocess,tomllib,urllib.request
config=json.loads(pathlib.Path('/etc/magnitude-lab-initialization.json').read_text())
expected={'x64':'x86_64','arm64':'aarch64'}[config['architecture']]
if platform.machine()!=expected:raise SystemExit('Initialization architecture mismatch')
account=pwd.getpwnam(config['adminUsername'])
if account.pw_uid==0:raise SystemExit('Worker account must not be root')
# Disposable worker authorization uses real pkexec/Polkit, limited to the packaged update command.
# The candidate still verifies its signed update and performs the native apt transaction.
update_program='/usr/lib/magnitude-desktop/resources/magnitude'
rule='polkit.addRule(function(action, subject) {\n' + \
 '  if (action.id === "org.freedesktop.policykit.exec" && subject.user === '+json.dumps(account.pw_name)+ \
 ' && action.lookup("program") === '+json.dumps(update_program)+ \
 ' && action.lookup("command_line").indexOf('+json.dumps(update_program+' _install-application-update ')+') === 0) return polkit.Result.YES;\n});\n'
policy=pathlib.Path('/etc/polkit-1/rules.d/49-magnitude-lab-update.rules')
policy.write_text(rule)
policy.chmod(0o644)
subprocess.run(['systemctl','start','polkit.service'],check=True)
home=pathlib.Path(account.pw_dir)
root=home/'lab-runtime'
root.mkdir(mode=0o700)
os.chown(root,account.pw_uid,account.pw_gid)
downloads=pathlib.Path('/opt/magnitude-lab-downloads')
downloads.mkdir(mode=0o700)
class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,*args,**kwargs):return None
opener=urllib.request.build_opener(NoRedirect)
def download(key,item,client):
 digest=hashlib.sha256(); total=0; destination=downloads/key
 try:
  with client.open(item['url'],timeout=120) as response,destination.open('xb') as out:
   for chunk in iter(lambda:response.read(1024*1024),b''):
    total+=len(chunk)
    if total>item['bytes']:raise ValueError('excess bytes')
    digest.update(chunk);out.write(chunk)
 except Exception:raise SystemExit('Pinned '+key+' download failed') from None
 if total!=item['bytes'] or digest.hexdigest()!=item['sha256']:raise SystemExit('Pinned '+key+' integrity mismatch')
for key in ['runtime','node','rustup']:download(key,config[key],opener)
# Extraction runs without root even though these are administrator-admitted archives.
for key in ['runtime','node','rustup']:
 shutil.move(str(downloads/key),str(root/('download-'+key)))
 os.chown(root/('download-'+key),account.pw_uid,account.pw_gid)
subprocess.run(['sudo','-n','-u',account.pw_name,'--','mkdir',str(root/'node-bin')],check=True)
subprocess.run(['sudo','-n','-u',account.pw_name,'--','tar','-xf',str(root/'download-node'),'--strip-components=1','-C',str(root/'node-bin')],check=True)
subprocess.run(['sudo','-n','-u',account.pw_name,'--','tar','-xzf',str(root/'download-runtime'),'-C',str(root)],check=True)
workspace=root/'runtime'
hermes=json.loads((workspace/'packages/testing-lab/tools/hermes.json').read_text())
if hermes['repository']!='https://github.com/NousResearch/hermes-agent.git' or not re.fullmatch('[a-f0-9]{40}',hermes['commit']):raise SystemExit('Invalid pinned Hermes source')
class HttpsRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,req,fp,code,msg,headers,newurl):
  if not newurl.startswith('https://'):raise SystemExit('Insecure native-tool redirect')
  return super().redirect_request(req,fp,code,msg,headers,newurl)
download('tirith',hermes['tirith']['downloads'][config['architecture']],urllib.request.build_opener(HttpsRedirect))
shutil.move(str(downloads/'tirith'),str(root/'download-tirith'))
os.chown(root/'download-tirith',account.pw_uid,account.pw_gid)
rust_version=tomllib.loads((workspace/'inference/rust-toolchain.toml').read_text())['toolchain']['channel']
path=f'{home}/.local/bin:{root}/node-bin/bin:{root}/tooling/node_modules/.bin:{home}/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
environment=[f'HOME={home}',f'USER={account.pw_name}',f'LOGNAME={account.pw_name}',f'PATH={path}',
 f'CARGO_HOME={home}/.cargo',f'RUSTUP_HOME={home}/.rustup',f'LAB_BUN_VERSION={config["bunVersion"]}',f'LAB_RUST_VERSION={rust_version}',
 f'LAB_HERMES_COMMIT={hermes["commit"]}',f'LAB_HERMES_VERSION={hermes["version"]}',f'LAB_TIRITH_VERSION={hermes["tirith"]["version"]}']
(root/'download-rustup').rename(root/'rustup-init')
os.chmod(root/'rustup-init',0o700)
subprocess.run(['sudo','-n','-u',account.pw_name,'--','env','-i',*environment,'/bin/bash','-c',r'''
set -euo pipefail
../rustup-init -y --profile minimal --default-toolchain "$LAB_RUST_VERSION"
npm install --prefix ../tooling --no-audit --no-fund "bun@$LAB_BUN_VERSION"
test "$(bun --version)" = "$LAB_BUN_VERSION"
bun install --frozen-lockfile --ignore-scripts
bun packages/version/scripts/generate-version.ts
npm ci --prefix packages/testing-lab/tools --no-audit --no-fund
python3 -m venv ../python-tools
../python-tools/bin/pip install --disable-pip-version-check --no-deps --require-hashes -r packages/testing-lab/tools/python-tools.txt
git init -q ../hermes-agent
git -C ../hermes-agent remote add origin https://github.com/NousResearch/hermes-agent.git
git -C ../hermes-agent fetch -q --depth=1 origin "$LAB_HERMES_COMMIT"
git -C ../hermes-agent checkout -q --detach FETCH_HEAD
test "$(git -C ../hermes-agent rev-parse HEAD)" = "$LAB_HERMES_COMMIT"
../python-tools/bin/uv sync --project ../hermes-agent --python /usr/bin/python3 --frozen --no-dev
mkdir ../native-tools
tar -xzf ../download-tirith -C ../native-tools tirith tirith-package-approval-authority
# The application intentionally excludes project-local node_modules/.bin from discovery.
# Expose this user's pinned installations through their normal executable directory.
mkdir -p "$HOME/.local/bin"
ln -s "$PWD/packages/testing-lab/tools/node_modules/.bin/pi" "$HOME/.local/bin/pi"
ln -s "$PWD/packages/testing-lab/tools/node_modules/.bin/opencode" "$HOME/.local/bin/opencode"
ln -s "$PWD/../hermes-agent/.venv/bin/hermes" "$HOME/.local/bin/hermes"
ln -s "$PWD/../native-tools/tirith" "$HOME/.local/bin/tirith"
ln -s "$PWD/../native-tools/tirith-package-approval-authority" "$HOME/.local/bin/tirith-package-approval-authority"
test "$(command -v pi)" = "$HOME/.local/bin/pi"
test "$(command -v opencode)" = "$HOME/.local/bin/opencode"
pi --version
opencode --version
hermes --version
test "$(tirith --version)" = "tirith $LAB_TIRITH_VERSION"
../hermes-agent/.venv/bin/python -c 'import importlib.metadata,os; assert importlib.metadata.version("hermes-agent")==os.environ["LAB_HERMES_VERSION"]'
bun -e 'await import("./packages/testing-lab/src/outward-worker.ts")'
'''],cwd=workspace,check=True)
# The root provisioning process remains private; package-building children need standard modes.
launcher='\n'.join(['#!/bin/bash','set -euo pipefail','umask 022',
 'export PATH='+shlex.quote(path),
 'export CARGO_HOME='+shlex.quote(str(home/'.cargo')),
 'export RUSTUP_HOME='+shlex.quote(str(home/'.rustup')),
 'export LAB_TERMINAL_NODE_EXECUTABLE='+shlex.quote(str(root/'node-bin/bin/node')),
 'export LAB_PI_EXECUTABLE='+shlex.quote(str(home/'.local/bin/pi')),
 'export LAB_OPENCODE_EXECUTABLE='+shlex.quote(str(home/'.local/bin/opencode')),
 'export LAB_HERMES_EXECUTABLE='+shlex.quote(str(home/'.local/bin/hermes')),
 'cd '+shlex.quote(str(workspace)),
 'exec xvfb-run -a -s "-screen 0 1600x1000x24" dbus-run-session -- /bin/bash /opt/magnitude-lab-display-worker',''])
pathlib.Path('/opt/magnitude-lab-worker').write_text(launcher)
os.chmod('/opt/magnitude-lab-worker',0o755)
receipt=pathlib.Path('/var/lib/magnitude-lab')
receipt.mkdir(mode=0o755,exist_ok=True)
(receipt/'runtime.json').write_text(json.dumps({'runtimeSha256':config['runtime']['sha256'],'nodeSha256':config['node']['sha256'],
 'rustupSha256':config['rustup']['sha256'],'architecture':config['architecture'],'bunVersion':config['bunVersion'],'rustVersion':rust_version}))
for key in ['runtime','node','tirith']:(root/('download-'+key)).unlink()
(root/'rustup-init').unlink()
PY
cat > /opt/magnitude-lab-display-worker <<'SCRIPT'
#!/bin/bash
set -euo pipefail
openbox >/tmp/magnitude-lab-openbox.log 2>&1 &
window_manager=$!
trap 'kill "$window_manager" 2>/dev/null || true; wait "$window_manager" 2>/dev/null || true' EXIT
bun packages/testing-lab/src/outward-worker.ts
SCRIPT
chmod 0755 /opt/magnitude-lab-display-worker
rm /etc/magnitude-lab-initialization.json
printf '%s\n' ready > /var/lib/magnitude-lab/ready
