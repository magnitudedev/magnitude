#!/bin/bash
set -euo pipefail
umask 077
# This file is installed by cloud-init before any run credential or candidate source arrives.
. /etc/os-release
python3 - "$ID" "$VERSION_ID" <<'CHECK'
import json,pathlib,sys
expected=json.loads(pathlib.Path('/etc/magnitude-lab-initialization.json').read_text())['distribution']
actual_os={'rhel':'redhat'}.get(sys.argv[1],sys.argv[1])
actual_version=sys.argv[2]
version_matches=actual_version==expected['version'] or (expected['os']=='redhat' and actual_version.startswith(expected['version']+'.'))
if expected['os']!=actual_os or not version_matches:raise SystemExit('Initialization distribution mismatch')
CHECK
case "$ID:$VERSION_ID" in
  ubuntu:24.04|debian:13)
    export DEBIAN_FRONTEND=noninteractive
    # Restarting the Azure agent while its readiness command runs can sever observation.
    export NEEDRESTART_MODE=l
    apt-get update -qq
    apt-get install -y -qq sudo curl ca-certificates python3 python3-venv git xz-utils tar build-essential cmake clang libclang-dev libssl-dev pkg-config fakeroot rpm binutils lsof nftables polkitd pkexec xvfb xauth dbus-x11 openbox libgtk-3-0t64 libnss3 libasound2t64 libgbm1 libxss1 libxtst6
    ;;
  fedora:44)
    dnf -y install sudo curl-minimal ca-certificates python3 python3-pip python3.13 python3.13-devel git xz tar gcc gcc-c++ make cmake clang clang-devel openssl-devel pkgconf-pkg-config fakeroot dpkg rpm-build binutils lsof nftables polkit xorg-x11-server-Xvfb xorg-x11-xauth dbus-x11 openbox gtk3 nss alsa-lib mesa-libgbm libXScrnSaver libXtst libffi-devel
    ;;
  rhel:10|rhel:10.*)
    # The Azure RHEL image leaves /home at 1 GiB even when its OS disk is larger.
    # Grow only the pinned image's existing home volume before installing tooling.
    python3 - <<'STORAGE'
import json,pathlib,subprocess
def output(*args):return subprocess.check_output(args,text=True).strip()
home=output('findmnt','-n','-o','SOURCE','-T','/home')
if output('findmnt','-n','-o','TARGET','-T','/home')!='/home' or output('findmnt','-n','-o','FSTYPE','-T','/home')!='xfs':
 raise SystemExit('Unexpected RHEL workspace filesystem')
volumes=json.loads(output('lvs','--reportformat','json','--units','b','--nosuffix','-o','lv_path,vg_name,lv_size',home))['report'][0]['lv']
if len(volumes)!=1 or volumes[0]['lv_path']!='/dev/rootvg/homelv':raise SystemExit('Unexpected RHEL workspace volume')
if float(volumes[0]['lv_size']) < 64*1024**3:
 physical=json.loads(output('pvs','--reportformat','json','-o','pv_name,vg_name'))['report'][0]['pv']
 physical=[p for p in physical if p['vg_name']==volumes[0]['vg_name']]
 if len(physical)!=1:raise SystemExit('Ambiguous RHEL workspace physical volume')
 partition=pathlib.Path(physical[0]['pv_name']).resolve()
 parent=output('lsblk','--nodeps','-n','-o','PKNAME',str(partition))
 number=(pathlib.Path('/sys/class/block')/partition.name/'partition').read_text().strip()
 if not parent or '/' in parent or '\n' in parent or not number.isdigit():raise SystemExit('Invalid RHEL workspace partition')
 growth=subprocess.run(['growpart','/dev/'+parent,number],capture_output=True,text=True)
 if growth.returncode!=0 and not (growth.returncode==1 and 'NOCHANGE:' in growth.stdout):
  raise SystemExit('Could not grow RHEL workspace partition: '+growth.stdout+growth.stderr)
 subprocess.run(['pvresize',str(partition)],check=True)
 subprocess.run(['lvextend','--resizefs','--size','64G',home],check=True)
if int(output('df','--output=avail','-B1','/home').splitlines()[-1]) < 48*1024**3:
 raise SystemExit('RHEL workspace has less than 48 GiB free')
STORAGE
    # RHEL consumers install RPMs built on the canonical Ubuntu producer. They do
    # not need an additional repository just to install Debian packaging tools.
    dnf -y install sudo curl ca-certificates python3 python3-pip python3-devel git xz tar gcc gcc-c++ make cmake clang clang-devel openssl-devel pkgconf-pkg-config rpm-build binutils lsof nftables polkit dbus-daemon gnome-shell gtk3 nss alsa-lib mesa-libgbm mesa-dri-drivers libXtst libffi-devel python3-gobject-base
    ;;
  *) printf '%s\n' 'Unsupported Linux worker distribution' >&2; exit 1 ;;
esac

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
 f'LAB_HERMES_PYTHON={"/usr/bin/python3.13" if config["distribution"]["os"] == "fedora" else "/usr/bin/python3"}',
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
../python-tools/bin/uv sync --project ../hermes-agent --python "$LAB_HERMES_PYTHON" --frozen --no-dev
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
display_command='exec dbus-run-session -- /bin/bash /opt/magnitude-lab-wayland-worker' if config['distribution']['os']=='redhat' else 'exec xvfb-run -a -s "-screen 0 1600x1000x24" dbus-run-session -- /bin/bash /opt/magnitude-lab-display-worker'
launcher='\n'.join(['#!/bin/bash','set -euo pipefail','umask 022',
 'export PATH='+shlex.quote(path),
 'export CARGO_HOME='+shlex.quote(str(home/'.cargo')),
 'export RUSTUP_HOME='+shlex.quote(str(home/'.rustup')),
 'export LAB_TERMINAL_NODE_EXECUTABLE='+shlex.quote(str(root/'node-bin/bin/node')),
 'export LAB_PI_EXECUTABLE='+shlex.quote(str(home/'.local/bin/pi')),
 'export LAB_OPENCODE_EXECUTABLE='+shlex.quote(str(home/'.local/bin/opencode')),
 'export LAB_HERMES_EXECUTABLE='+shlex.quote(str(home/'.local/bin/hermes')),
 'cd '+shlex.quote(str(workspace)),
 display_command,''])
pathlib.Path('/opt/magnitude-lab-worker').write_text(launcher)
os.chmod('/opt/magnitude-lab-worker',0o755)
receipt=pathlib.Path('/var/lib/magnitude-lab')
receipt.mkdir(mode=0o755,exist_ok=True)
(receipt/'runtime.json').write_text(json.dumps({'runtimeSha256':config['runtime']['sha256'],'nodeSha256':config['node']['sha256'],
 'rustupSha256':config['rustup']['sha256'],'architecture':config['architecture'],'distribution':config['distribution'],'bunVersion':config['bunVersion'],'rustVersion':rust_version}))
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
cat > /opt/magnitude-lab-wayland-worker <<'SCRIPT'
#!/bin/bash
set -euo pipefail
export XDG_RUNTIME_DIR
XDG_RUNTIME_DIR=$(mktemp -d /tmp/magnitude-lab-wayland.XXXXXXXX)
chmod 0700 "$XDG_RUNTIME_DIR"
export XDG_SESSION_TYPE=wayland
export WAYLAND_DISPLAY=wayland-lab
unset DISPLAY
compositor=''
cleanup() {
  status=$?
  if [ "$status" -ne 0 ] && [ -f "$XDG_RUNTIME_DIR/compositor.log" ]; then
    tail -c 16384 "$XDG_RUNTIME_DIR/compositor.log" >&2
  fi
  if [ -n "$compositor" ]; then
    kill "$compositor" 2>/dev/null || true
    wait "$compositor" 2>/dev/null || true
  fi
  rm -rf -- "$XDG_RUNTIME_DIR"
}
trap cleanup EXIT
# Software composition affects the test display, not the inference backend.
LIBGL_ALWAYS_SOFTWARE=1 gnome-shell --wayland --headless --virtual-monitor 1600x1000 --wayland-display "$WAYLAND_DISPLAY" >"$XDG_RUNTIME_DIR/compositor.log" 2>&1 &
compositor=$!
export LAB_COMPOSITOR_PID="$compositor"
python3 - <<'READY'
import os,pathlib,time
from gi.repository import Gio
deadline=time.monotonic()+60
socket=pathlib.Path(os.environ['XDG_RUNTIME_DIR'])/os.environ['WAYLAND_DISPLAY']
while time.monotonic()<deadline:
 os.kill(int(os.environ['LAB_COMPOSITOR_PID']),0)
 try:
  bus=Gio.bus_get_sync(Gio.BusType.SESSION,None)
  state=bus.call_sync('org.gnome.Mutter.DisplayConfig','/org/gnome/Mutter/DisplayConfig','org.gnome.Mutter.DisplayConfig','GetCurrentState',None,None,Gio.DBusCallFlags.NONE,1000,None).unpack()
  if socket.is_socket() and len(state[2])>0:break
 except Exception:pass
 time.sleep(0.25)
else:raise SystemExit('Wayland compositor did not publish a logical monitor')
READY
bun packages/testing-lab/src/outward-worker.ts
SCRIPT
chmod 0755 /opt/magnitude-lab-wayland-worker
rm /etc/magnitude-lab-initialization.json
printf '%s\n' ready > /var/lib/magnitude-lab/ready
