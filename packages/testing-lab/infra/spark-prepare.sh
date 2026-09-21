#!/bin/bash
set -euo pipefail
test "$(uname -m)" = aarch64
curl -fsSL https://nodejs.org/dist/v24.21.0/node-v24.21.0-linux-arm64.tar.xz -o node.tar.xz
echo '6ad1325edbdb5649c379b75a237147a666c95d4f9ae8d340fef2d1575d289ad2  node.tar.xz' | sha256sum -c -
mkdir node
tar -xJf node.tar.xz --strip-components=1 -C node
rm node.tar.xz
tar -xzf runtime.tar.gz
cd runtime
npm install --prefix ../tooling --no-audit --no-fund bun@1.4.2
test "$(bun --version)" = 1.4.2
bun install --frozen-lockfile --ignore-scripts
bun packages/version/scripts/generate-version.ts
npm ci --prefix packages/testing-lab/tools --no-audit --no-fund
python3 -m venv ../python-tools
../python-tools/bin/pip install --disable-pip-version-check --no-deps --require-hashes -r packages/testing-lab/tools/python-tools.txt
python3 - <<'PY'
import hashlib,json,pathlib,subprocess,urllib.request
pins=json.loads(pathlib.Path('packages/testing-lab/tools/hermes.json').read_text())
root=pathlib.Path('/opt/lab')
subprocess.run(['git','init','-q',str(root/'hermes-agent')],check=True)
subprocess.run(['git','-C',str(root/'hermes-agent'),'remote','add','origin',pins['repository']],check=True)
subprocess.run(['git','-C',str(root/'hermes-agent'),'fetch','-q','--depth=1','origin',pins['commit']],check=True)
subprocess.run(['git','-C',str(root/'hermes-agent'),'checkout','-q','--detach','FETCH_HEAD'],check=True)
if subprocess.check_output(['git','-C',str(root/'hermes-agent'),'rev-parse','HEAD'],text=True).strip()!=pins['commit']:raise SystemExit('Hermes source mismatch')
subprocess.run([str(root/'python-tools/bin/uv'),'sync','--project',str(root/'hermes-agent'),'--python','/usr/bin/python3','--frozen','--no-dev'],check=True)
artifact=pins['tirith']['downloads']['arm64']
data=urllib.request.urlopen(artifact['url'],timeout=120).read(artifact['bytes']+1)
if len(data)!=artifact['bytes'] or hashlib.sha256(data).hexdigest()!=artifact['sha256']:raise SystemExit('Tirith integrity mismatch')
(root/'tirith.tar.gz').write_bytes(data)
(root/'native-tools').mkdir()
subprocess.run(['tar','-xzf',str(root/'tirith.tar.gz'),'-C',str(root/'native-tools'),'tirith','tirith-package-approval-authority'],check=True)
(root/'tirith.tar.gz').unlink()
PY
mkdir -p "$HOME/.local/bin"
ln -s "$PWD/packages/testing-lab/tools/node_modules/.bin/pi" "$HOME/.local/bin/pi"
ln -s "$PWD/packages/testing-lab/tools/node_modules/.bin/opencode" "$HOME/.local/bin/opencode"
ln -s /opt/lab/hermes-agent/.venv/bin/hermes "$HOME/.local/bin/hermes"
ln -s /opt/lab/native-tools/tirith "$HOME/.local/bin/tirith"
ln -s /opt/lab/native-tools/tirith-package-approval-authority "$HOME/.local/bin/tirith-package-approval-authority"
pi --version
opencode --version
hermes --version
bun -e 'await import("./packages/testing-lab/src/worker-entry.ts")'
