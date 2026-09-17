// Real old-release -> packaged desktop acceptance. Run in an isolated home with port 10100 free.
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createServer } from 'node:net';
import { mkdir, writeFile, readFile, open, readdir, stat } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { join, resolve } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';

const home = resolve(process.env.MAGNITUDE_UPGRADE_TEST_HOME ?? '');
const oldExecutable = process.env.MAGNITUDE_UPGRADE_OLD_EXECUTABLE;
const desktopExecutable = process.env.MAGNITUDE_TEST_DESKTOP_EXECUTABLE;
assert(home.includes('magnitude-upgrade-validation') && oldExecutable && desktopExecutable);
const environment = { ...process.env, HOME: home, PATH: '/usr/bin:/bin:/usr/sbin:/sbin', MAGNITUDE_SHELL_ENV_INHERITED: '1' };
delete environment.MAGNITUDE_DEV_DATA_DIR;
delete environment.MAGNITUDE_DEV_PORT;
delete environment.MAGNITUDE_DESKTOP_STATE_DIR;
const root = join(home, '.magnitude');
const sentinels = ['models/upgrade-preserved.bin', 'sessions/upgrade-preserved.txt', 'upgrade-preserved.txt'];
for (const path of sentinels) { await mkdir(join(root, path, '..'), { recursive: true }); await writeFile(join(root, path), 'preserve-me'); }
if (!process.env.MAGNITUDE_UPGRADE_EXISTING_SERVICE) await new Promise((resolve, reject) => { const s = createServer(); s.once('error', reject); s.listen(10100, '127.0.0.1', () => s.close(resolve)); });
const health = async () => { try { return await (await fetch('http://127.0.0.1:10100/health', { signal: AbortSignal.timeout(1000) })).json(); } catch { return null; } };
const until = async (read, description) => { const deadline = Date.now() + 180000; do { if (await read()) return; await delay(100); } while (Date.now() < deadline); throw new Error(`Timed out: ${description}`); };
const alive = pid => { try { process.kill(pid, 0); return true; } catch (e) { if (e.code === 'ESRCH') return false; throw e; } };
const oldLog = await open(join(home, 'old-packaged-upgrade.log'), 'w');
const newLog = await open(join(home, 'new-packaged-upgrade.log'), 'w');
const modelSnapshot = async () => {
  const result = {};
  const visit = async directory => {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      const path = join(directory, entry.name);
      if (entry.isDirectory()) await visit(path);
      else if (entry.isFile() && directory.endsWith('/blobs')) {
        const hash = createHash('sha256'); for await (const chunk of createReadStream(path)) hash.update(chunk);
        const metadata = await stat(path); result[path] = { hash: hash.digest('hex'), size: metadata.size, mtime: metadata.mtimeMs, inode: metadata.ino };
      }
    }
  };
  await visit(join(root, 'models')); return result;
};
const beforeModels = await modelSnapshot();
let old, app;
let oldPid;
try {
  if (!process.env.MAGNITUDE_UPGRADE_EXISTING_SERVICE) old = spawn(oldExecutable, ['serve', '--data-dir', root], { env: environment, detached: true, stdio: ['ignore', oldLog.fd, oldLog.fd] });
  await until(async () => (await health())?.version === '0.0.15', 'old release ready');
  oldPid = (await health()).pid;
  console.log('Old release serving:', await health());
  app = spawn(desktopExecutable, ['--background'], { env: environment, detached: true, stdio: ['ignore', newLog.fd, newLog.fd] });
  await until(async () => { const current = await health(); return current?.pid !== oldPid && current?.state?._tag === 'Ready'; }, 'new packaged app ready');
  assert(!alive(oldPid), 'old service must exit');
  for (const path of sentinels) assert.equal(await readFile(join(root, path), 'utf8'), 'preserve-me');
  console.log('New packaged app ready; old process gone; data preserved:', await health());
  assert.deepEqual(await modelSnapshot(), beforeModels, 'model blobs must remain identical and untouched');
  if (process.env.MAGNITUDE_UPGRADE_TEST_MODEL) {
    assert(Object.keys(beforeModels).length > 0, 'real downloaded model required');
    const response = await fetch('http://127.0.0.1:10100/inference/v1/chat/completions', {
      method: 'POST', headers: { 'content-type': 'application/json' }, signal: AbortSignal.timeout(180000),
      body: JSON.stringify({ model: process.env.MAGNITUDE_UPGRADE_TEST_MODEL, messages: [{ role: 'user', content: 'What is 2 + 2? Answer briefly.' }], max_tokens: 32, stream: false }),
    });
    const body = await response.text(); assert.equal(response.status, 200, body); assert(JSON.parse(body).choices?.length > 0 && JSON.parse(body).usage?.completion_tokens > 0, body);
    assert.deepEqual(await modelSnapshot(), beforeModels, 'inference must reuse existing model blobs');
    console.log('Existing downloaded model inference succeeded:', body);
  }
  app.kill('SIGTERM'); await until(() => !alive(app.pid), 'new app quit');
  app = spawn(desktopExecutable, ['--background'], { env: environment, detached: true, stdio: ['ignore', newLog.fd, newLog.fd] });
  await until(async () => (await health())?.state?._tag === 'Ready', 'repeat launch');
  console.log('Repeat packaged launch ready');
} finally {
  if (app && alive(app.pid)) { app.kill('SIGTERM'); await until(() => !alive(app.pid), 'cleanup app'); }
  if (old && alive(old.pid)) { old.kill('SIGTERM'); await until(() => !alive(old.pid), 'cleanup old service'); }
  await oldLog.close(); await newLog.close();
}
