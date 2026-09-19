#!/bin/bash
set -euo pipefail
umask 077
# This file is installed by cloud-init before any run credential or candidate source arrives.
. /etc/os-release
test "$ID" = ubuntu && test "$VERSION_ID" = 24.04
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq curl ca-certificates python3 git xz-utils tar build-essential cmake libclang-dev libssl-dev pkg-config fakeroot rpm binutils nftables xvfb xauth dbus-x11 openbox libgtk-3-0t64 libnss3 libasound2t64 libgbm1 libxss1 libxtst6

python3 - <<'PY'
import hashlib,json,os,pathlib,platform,pwd,shlex,shutil,subprocess,tomllib,urllib.request
config=json.loads(pathlib.Path('/etc/magnitude-lab-initialization.json').read_text())
expected={'x64':'x86_64','arm64':'aarch64'}[config['architecture']]
if platform.machine()!=expected:raise SystemExit('Initialization architecture mismatch')
account=pwd.getpwnam(config['adminUsername'])
if account.pw_uid==0:raise SystemExit('Worker account must not be root')
home=pathlib.Path(account.pw_dir)
root=home/'lab-runtime'
root.mkdir(mode=0o700)
os.chown(root,account.pw_uid,account.pw_gid)
downloads=pathlib.Path('/opt/magnitude-lab-downloads')
downloads.mkdir(mode=0o700)
class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,*args,**kwargs):return None
opener=urllib.request.build_opener(NoRedirect)
for key in ['runtime','node','rustup']:
 item=config[key]; digest=hashlib.sha256(); total=0; destination=downloads/key
 try:
  with opener.open(item['url'],timeout=120) as response,destination.open('xb') as out:
   for chunk in iter(lambda:response.read(1024*1024),b''):
    total+=len(chunk)
    if total>item['bytes']:raise ValueError('excess bytes')
    digest.update(chunk);out.write(chunk)
 except Exception:raise SystemExit('Pinned '+key+' download failed') from None
 if total!=item['bytes'] or digest.hexdigest()!=item['sha256']:raise SystemExit('Pinned '+key+' integrity mismatch')
# Extraction runs without root even though these are administrator-admitted archives.
for key in ['runtime','node','rustup']:
 shutil.move(str(downloads/key),str(root/('download-'+key)))
 os.chown(root/('download-'+key),account.pw_uid,account.pw_gid)
subprocess.run(['sudo','-n','-u',account.pw_name,'--','mkdir',str(root/'node-bin')],check=True)
subprocess.run(['sudo','-n','-u',account.pw_name,'--','tar','-xf',str(root/'download-node'),'--strip-components=1','-C',str(root/'node-bin')],check=True)
subprocess.run(['sudo','-n','-u',account.pw_name,'--','tar','-xzf',str(root/'download-runtime'),'-C',str(root)],check=True)
workspace=root/'runtime'
rust_version=tomllib.loads((workspace/'inference/rust-toolchain.toml').read_text())['toolchain']['channel']
path=f'{root}/node-bin/bin:{root}/tooling/node_modules/.bin:{workspace}/packages/testing-lab/tools/node_modules/.bin:{home}/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
environment=[f'HOME={home}',f'USER={account.pw_name}',f'LOGNAME={account.pw_name}',f'PATH={path}',
 f'CARGO_HOME={home}/.cargo',f'RUSTUP_HOME={home}/.rustup',f'LAB_BUN_VERSION={config["bunVersion"]}',f'LAB_RUST_VERSION={rust_version}']
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
test "$(command -v pi)" = "$PWD/packages/testing-lab/tools/node_modules/.bin/pi"
test "$(command -v opencode)" = "$PWD/packages/testing-lab/tools/node_modules/.bin/opencode"
pi --version
opencode --version
bun -e 'await import("./packages/testing-lab/src/outward-worker.ts")'
'''],cwd=workspace,check=True)
launcher='\n'.join(['#!/bin/bash','set -euo pipefail','umask 077',
 'export PATH='+shlex.quote(path),
 'export CARGO_HOME='+shlex.quote(str(home/'.cargo')),
 'export RUSTUP_HOME='+shlex.quote(str(home/'.rustup')),
 'export LAB_TERMINAL_NODE_EXECUTABLE='+shlex.quote(str(root/'node-bin/bin/node')),
 'export LAB_PI_EXECUTABLE='+shlex.quote(str(workspace/'packages/testing-lab/tools/node_modules/.bin/pi')),
 'export LAB_OPENCODE_EXECUTABLE='+shlex.quote(str(workspace/'packages/testing-lab/tools/node_modules/.bin/opencode')),
 'cd '+shlex.quote(str(workspace)),
 'exec xvfb-run -a -s "-screen 0 1600x1000x24" dbus-run-session -- /bin/bash /opt/magnitude-lab-display-worker',''])
pathlib.Path('/opt/magnitude-lab-worker').write_text(launcher)
os.chmod('/opt/magnitude-lab-worker',0o755)
receipt=pathlib.Path('/var/lib/magnitude-lab')
receipt.mkdir(mode=0o755)
(receipt/'runtime.json').write_text(json.dumps({'runtimeSha256':config['runtime']['sha256'],'nodeSha256':config['node']['sha256'],
 'rustupSha256':config['rustup']['sha256'],'architecture':config['architecture'],'bunVersion':config['bunVersion'],'rustVersion':rust_version}))
for key in ['runtime','node']:(root/('download-'+key)).unlink()
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
