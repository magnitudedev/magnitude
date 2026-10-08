import { _electron as electron } from 'playwright';
import assert from 'node:assert/strict';
import { mkdtemp, rm, readFile, writeFile, chmod, mkdir } from 'node:fs/promises';
import { createServer } from 'node:http';
import { createConnection } from 'node:net';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { spawn, spawnSync, execFileSync } from 'node:child_process';
import { setTimeout as delay } from 'node:timers/promises';

// Playwright's Electron driver runs under Node; the owning Vitest suite runs under Bun.
// Preserve normal Chromium launch flags: Playwright otherwise injects --no-sandbox.
// This does not override the application's BrowserWindow preload sandbox policy.
assert.ok(!process.versions.bun, 'Set MAGNITUDE_TEST_NODE to an absolute Node executable; Bun supplies a node shim on PATH.');
const executablePath = process.env.MAGNITUDE_TEST_DESKTOP_EXECUTABLE;
assert.ok(executablePath);
const profile = await mkdtemp(join(tmpdir(), 'magnitude-app-test-'));
const failedProfile = await mkdtemp('/tmp/mag-failed-');
const probeShell = join(profile, 'slow-shell');
const probePids = join(profile, 'shell-pids');
await writeFile(probeShell, '#!/bin/sh\necho "$PPID" >> "$MAGNITUDE_TEST_SHELL_PIDS"\necho $$ >> "$MAGNITUDE_TEST_SHELL_PIDS"\ntrap "" TERM\nwhile :; do /bin/sleep 30 & echo $! >> "$MAGNITUDE_TEST_SHELL_PIDS"; wait; done\n');
await chmod(probeShell, 0o700);
const env = { ...process.env, SHELL: probeShell, MAGNITUDE_TEST_SHELL_PIDS: probePids, MAGNITUDE_DEV_DATA_DIR: profile, MAGNITUDE_DEV_PORT: '11109' };
delete env.MAGNITUDE_SHELL_ENV_INHERITED;
const control = (dataDirectory, intent) => new Promise((resolve, reject) => {
  const socket = createConnection(join(dataDirectory, 'state/application.sock'));
  let data = '';
  socket.setTimeout(5000, () => socket.destroy(new Error(`Application control ${intent} timed out`)));
  socket.on('error', reject);
  socket.on('connect', () => socket.write(JSON.stringify({ version: 1, intent }) + '\n'));
  socket.on('data', chunk => {
    data += chunk;
    if (!data.includes('\n')) return;
    socket.destroy();
    try { resolve(JSON.parse(data.split('\n')[0])); } catch (error) { reject(error); }
  });
});
const alive = pid => {
  try { process.kill(pid, 0); return true; }
  catch (error) { if (error.code === 'ESRCH') return false; throw error; }
};
const eventually = async (read, expected, timeout = 30000) => {
  const deadline = Date.now() + timeout;
  let actual;
  do {
    actual = await read();
    if (JSON.stringify(actual) === JSON.stringify(expected)) return;
    await delay(100);
  } while (Date.now() < deadline);
  assert.deepEqual(actual, expected, `Condition did not settle within ${timeout} ms; application output:\n${ownerOutput}`);
};
// The owning application's recent main-process output, reported when a contender stalls.
let ownerOutput = '';
const launchOwner = async options => {
  const launched = await electron.launch(options);
  ownerOutput = '';
  const collect = chunk => { ownerOutput = (ownerOutput + chunk.toString()).slice(-32000); };
  launched.process().stdout.on('data', collect);
  launched.process().stderr.on('data', collect);
  return launched;
};
// Native stacks show whether a stalled process waits on the control socket or is busy.
const sampleStack = pid => {
  try { return execFileSync('/usr/bin/sample', [String(pid), '2'], { encoding: 'utf8', timeout: 20000, maxBuffer: 16 * 1024 * 1024 }).slice(0, 16000); }
  catch (error) { return `sample failed: ${error.message}`; }
};
// A contender cold-starts Electron, then makes two handoff exchanges (Observe, then its intent),
// each of which the application allows 5 seconds (application-control.ts).
const contenderLimit = 60000;
const invoke = args => new Promise((resolve, reject) => {
  const started = Date.now();
  const child = spawn(executablePath, args, { env, stdio: ['ignore', 'pipe', 'pipe'] });
  let output = '';
  const collect = chunk => { output = (output + chunk.toString()).slice(-32000); };
  child.stdout.on('data', collect); child.stderr.on('data', collect);
  const timeout = setTimeout(() => {
    const owner = application?.process();
    const report = [
      `Application contender [${args.join(' ')}] did not exit within ${contenderLimit} ms.`,
      `Contender output:\n${output}`,
      `Contender stack:\n${sampleStack(child.pid)}`,
      ...(owner ? [`Owner stack:\n${sampleStack(owner.pid)}`, `Owner output:\n${ownerOutput}`] : []),
    ].join('\n\n');
    child.kill('SIGKILL');
    reject(new Error(report));
  }, contenderLimit);
  child.once('error', error => { clearTimeout(timeout); reject(error); });
  child.once('exit', code => {
    clearTimeout(timeout);
    console.log(`Application contender [${args.join(' ')}] exited ${code} after ${Date.now() - started} ms`);
    resolve(code);
  });
});
let application;
let probeOwner;
let incumbent;
try {
  const invalidStateDirectory = join(profile, 'not-a-directory');
  await writeFile(invalidStateDirectory, 'deliberately invalid isolated ownership location');
  const rejected = await new Promise((resolve, reject) => {
    const child = spawn(executablePath, ['--background'], {
      env: { ...env, MAGNITUDE_DESKTOP_STATE_DIR: invalidStateDirectory },
      detached: true, stdio: ['ignore', 'pipe', 'pipe'],
    });
    let output = '';
    const collect = chunk => { output = (output + chunk.toString()).slice(-32000); };
    child.stdout.on('data', collect); child.stderr.on('data', collect);
    const timeout = setTimeout(() => {
      // This fixture owns the freshly spawned group, including Chromium helpers.
      try { process.kill(-child.pid, 'SIGKILL'); } catch {}
      reject(new Error(`Failed startup did not exit: ${output}`));
    }, 30000);
    child.once('error', error => { clearTimeout(timeout); reject(error); });
    child.once('exit', code => { clearTimeout(timeout); resolve({ code, output }); });
  });
  assert.equal(rejected.code, 1, rejected.output);
  assert.match(rejected.output, /ApplicationOwnershipFailed|EEXIST/);
  assert.doesNotMatch(rejected.output, /Cause\.reduceWithContext|UnhandledPromiseRejection/);
  console.log('Rejected ownership path: original failure reported and background process exited');
  application = await launchOwner({ chromiumSandbox: true, executablePath, args: [], env: {
    ...env, MAGNITUDE_DEV_DATA_DIR: failedProfile,
    MAGNITUDE_ICN_PATH: join(failedProfile, 'absent-engine.json'),
  }, timeout: 60000 });
  const failedOwner = application.process();
  await eventually(async () => (await control(failedProfile, 'Observe').catch(() => null))?.service?._tag, 'Failed', 60000);
  for (let attempt = 0; attempt < 2; attempt++) {
    const snapshot = await control(failedProfile, 'Observe');
    const failure = snapshot.service;
    assert.equal(failure._tag, 'Failed');
    assert.ok(failure.message, 'Failed service must report its cause');
    assert.equal(snapshot.owner.tray._tag, 'Registered');
    assert.equal(alive(failedOwner.pid), true);
    if (attempt === 0) {
      await control(failedProfile, 'Retry');
      await eventually(async () => (await control(failedProfile, 'Observe')).service._tag, 'Starting', 30000);
      await eventually(async () => (await control(failedProfile, 'Observe')).service._tag, 'Failed', 60000);
    }
  }
  const failedQuit = application.waitForEvent('close', { timeout: 30000 });
  await control(failedProfile, 'Quit');
  await failedQuit;
  application = undefined;
  assert.equal(failedOwner.exitCode, 0);
  console.log('Missing inference engine: startup and retry report failure through application control; owner survives and Quit exits0');

  await mkdir(join(profile, 'acn'), { recursive: true });
  const oldState = join(profile, 'acn/coordination.sqlite');
  await writeFile(oldState, 'old coordination state: deliberately not a database');
  await writeFile(join(profile, 'preserved.txt'), 'preserve user data');
  incumbent = createServer((_request, response) => response.end('unrelated service'));
  await new Promise((resolve, reject) => { incumbent.once('error', reject); incumbent.listen(11109, '127.0.0.1', resolve); });
  application = await launchOwner({ chromiumSandbox: true, executablePath, args: ['--background'], env, timeout: 60000 });
  const app = application;
  const visibility = () => app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows().map(window => ({ visible: window.isVisible(), minimized: window.isMinimized() })));
  const health = () => fetch('http://127.0.0.1:11109/health').then(response => response.json()).catch(() => null);
  await eventually(visibility, [{ visible: false, minimized: false }]);
  await eventually(() => readFile(probePids, 'utf8').then(text => text.trim().split('\n').map(Number).some(alive)).catch(() => false), true);
  console.log('Slow shell probe: owner and hidden window available while shell is still running');
  assert.equal(await invoke([]), 0);
  await eventually(async () => (await control(profile, 'Observe').catch(() => null))?.service?._tag, 'Failed', 60000);
  assert.equal((await control(profile, 'Observe')).owner.tray._tag, 'Registered');
  assert.equal(await (await fetch('http://127.0.0.1:11109')).text(), 'unrelated service');
  await control(profile, 'Retry');
  await eventually(async () => (await control(profile, 'Observe')).service._tag, 'Starting', 30000);
  await eventually(async () => (await control(profile, 'Observe')).service._tag, 'Failed', 60000);
  assert.equal(await (await fetch('http://127.0.0.1:11109')).text(), 'unrelated service');
  await new Promise(resolve => incumbent.close(resolve));
  incumbent = undefined;
  await control(profile, 'Retry');
  await eventually(async () => (await control(profile, 'Observe')).service._tag, 'Starting', 30000);
  await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].close());
  console.log('Port conflict: control reports failure, owner stays available, incumbent survives retry');
  await eventually(async () => (await health())?.state?._tag, 'Ready', 60000);
  await eventually(visibility, [{ visible: false, minimized: false }]);
  const service = await health();
  if (process.env.MAGNITUDE_TEST_EXPECT_VERSION) assert.equal(service.version, process.env.MAGNITUDE_TEST_EXPECT_VERSION);
  if (process.env.MAGNITUDE_TEST_EXPECT_REVISION) assert.equal(service.revision, Number(process.env.MAGNITUDE_TEST_EXPECT_REVISION));
  if (process.env.MAGNITUDE_TEST_EXPECT_RPC_VERSION) assert.equal(service.rpcVersion, Number(process.env.MAGNITUDE_TEST_EXPECT_RPC_VERSION));
  await assert.rejects(readFile(join(profile, 'desktop/legacy-migration.json')), { code: 'ENOENT' });
  assert.equal(await readFile(oldState, 'utf8'), 'old coordination state: deliberately not a database');
  assert.equal(await readFile(join(profile, 'preserved.txt'), 'utf8'), 'preserve user data');
  console.log('Clean cutover: old coordination files untouched, no migration checkpoint, control Retry reaches Ready');
  console.log('Cold background launch: service Ready, window hidden');

  assert.deepEqual(await Promise.all(Array.from({ length: 4 }, () => invoke(['--background']))), [0, 0, 0, 0]);
  assert.equal((await health()).pid, service.pid);
  assert.deepEqual(await visibility(), [{ visible: false, minimized: false }]);
  console.log('Four concurrent background launches: same service, window hidden');

  assert.equal(await invoke([]), 0);
  await eventually(visibility, [{ visible: true, minimized: false }]);
  const window = await app.firstWindow();
  window.setDefaultTimeout(10000);
  // Connections go through the service, so the packaged CLI reaches the same failure the app shows.
  const rejectedCommand = spawnSync(join(dirname(dirname(executablePath)), 'Resources', 'magnitude'), ['connections', 'add', 'codex'], { env, encoding: 'utf8' });
  assert.notEqual(rejectedCommand.status, 0);
  const rejectedConnection = `${rejectedCommand.stdout}${rejectedCommand.stderr}`;
  assert.match(rejectedConnection, /No installed Magnitude models are available|Codex is not installed/);
  assert.doesNotMatch(rejectedConnection, /UnknownException|FiberFailure|Effect\.tryPromise|\n\s+at /);
  console.log('Rejected connection preserves actionable host failure and does not create a managed connection');
  await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].close());
  await eventually(visibility, [{ visible: false, minimized: false }]);
  assert.equal((await health()).pid, service.pid);
  assert.equal(await invoke(['--background']), 0);
  assert.deepEqual(await visibility(), [{ visible: false, minimized: false }]);
  console.log('Window Close: window stays hidden, service identity is retained, background demand does not reopen it');

  // Native event simulation; physical Dock and tray interactions are additional CUA acceptance.
  await app.evaluate(({ app }) => app.emit('activate'));
  await eventually(visibility, [{ visible: true, minimized: false }]);
  await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].minimize());
  await eventually(async () => (await visibility())[0]?.minimized, true);
  assert.equal(await invoke([]), 0);
  await eventually(visibility, [{ visible: true, minimized: false }]);
  console.log('Dock activation event and Open restore hidden/minimized windows');

  await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].close());
  const rendererStatus = () => app.evaluate(async ({ BrowserWindow }) => {
    const contents = BrowserWindow.getAllWindows()[0].webContents;
    const status = { crashed: contents.isCrashed(), loading: contents.isLoading(), documentReady: false };
    if (status.crashed || status.loading) return status;
    return Promise.race([
      contents.executeJavaScript('document.readyState === "complete" && !!window.__magnitudeDesktop')
        .then(documentReady => ({ ...status, documentReady })),
      new Promise(resolve => setTimeout(() => resolve(status), 1000)),
    ]);
  }).catch(error => ({ error: String(error) }));
  const rendererReady = () => rendererStatus().then(status => status.documentReady === true);
  const waitForRenderer = async () => {
    try { await eventually(rendererReady, true); }
    catch (error) { throw new Error(`Renderer did not recover: ${JSON.stringify(await rendererStatus())}\n${ownerOutput}`, { cause: error }); }
  };
  const crashRenderer = () => app.evaluate(async ({ BrowserWindow }) => {
    const contents = BrowserWindow.getAllWindows()[0].webContents;
    const gone = new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        contents.removeListener('render-process-gone', onGone);
        reject(new Error('Renderer crash event was not observed within 30 seconds'));
      }, 30000);
      const onGone = () => { clearTimeout(timer); resolve(); };
      contents.once('render-process-gone', onGone);
    });
    contents.forcefullyCrashRenderer();
    await gone;
  });
  await waitForRenderer();
  for (let attempt = 0; attempt < 3; attempt++) {
    console.log(`Crashing renderer: attempt ${attempt + 1}`);
    await crashRenderer();
    await waitForRenderer();
    console.log(`Renderer recovered: attempt ${attempt + 1}`);
    assert.equal((await health()).pid, service.pid);
    assert.deepEqual(await visibility(), [{ visible: false, minimized: false }]);
  }
  await crashRenderer();
  await delay(1000);
  assert.equal(await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].webContents.isCrashed()), true);
  assert.equal(await invoke(['--background']), 0);
  assert.equal(await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].webContents.isCrashed()), true);
  assert.equal((await health()).pid, service.pid);
  assert.equal(await invoke([]), 0);
  await waitForRenderer();
  assert.deepEqual(await visibility(), [{ visible: true, minimized: false }]);
  console.log('Renderer crashes: three bounded retries, background demand cannot renew them, explicit Open recovers the same service');

  const ownerQuit = app.waitForEvent('close', { timeout: 30000 });
  await control(profile, 'Quit');
  await ownerQuit;
  application = undefined;
  assert.equal(await health(), null);
  assert.throws(() => process.kill(service.pid, 0));
  await eventually(() => readFile(probePids, 'utf8').then(text => text.trim().split('\n').map(Number).some(alive)), false);
  console.log('Application control Quit: service and shell probe groups absent, endpoint closed');

  // Probe cleanup and inference readiness have independent deadlines. Observe the
  // probe directly so a slow engine boot cannot make this crash case miss it.
  await writeFile(probePids, '');
  probeOwner = spawn(executablePath, ['--background'], { env, stdio: 'ignore' });
  let activeProbe = [];
  await eventually(async () => {
    activeProbe = (await readFile(probePids, 'utf8')).trim().split('\n').map(Number).filter(pid => pid > 0);
    return activeProbe.length >= 3 && activeProbe.every(alive);
  }, true);
  const probeOwnerExited = new Promise((resolve, reject) => {
    probeOwner.once('exit', resolve);
    probeOwner.once('error', reject);
  });
  probeOwner.kill('SIGKILL');
  await probeOwnerExited;
  probeOwner = undefined;
  await eventually(() => Promise.resolve(activeProbe.some(alive)), false);
  await eventually(health, null);
  console.log('Forced owner death during shell discovery: observed live probe and descendant retired');

  application = await launchOwner({ chromiumSandbox: true, executablePath, args: ['--background'], env, timeout: 60000 });
  await eventually(async () => (await health())?.state?._tag, 'Ready', 60000);
  assert.deepEqual(await application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows().map(window => window.isVisible())), [false]);
  console.log('Full Quit and hidden relaunch preserve background window state');
  const replacement = await health();
  const descendants = execFileSync('/bin/ps', ['-axo', 'pid=,ppid='], { encoding: 'utf8' }).trim().split('\n')
    .map(line => line.trim().split(/\s+/).map(Number)).filter(([, parent]) => parent === replacement.pid).map(([pid]) => pid);
  assert.ok(descendants.length > 0, 'Ready service owns its inference child');
  const crashed = application.waitForEvent('close');
  application.process().kill('SIGKILL');
  await crashed;
  application = undefined;
  await eventually(() => Promise.resolve([replacement.pid, ...descendants].some(alive)), false);
  assert.equal(await health(), null);
  await eventually(() => readFile(probePids, 'utf8').then(text => text.trim().split('\n').map(Number).some(alive)), false);
  console.log('Forced owner death after Ready: service and inference removed by lifetime guards');

  application = await launchOwner({ chromiumSandbox: true, executablePath, args: ['--background'], env, timeout: 60000 });
  await eventually(async () => (await health())?.state?._tag, 'Ready', 60000);
  assert.notEqual((await health()).pid, replacement.pid);
  const terminatedService = (await health()).pid;
  const terminatedProcess = application.process();
  const terminated = application.waitForEvent('close', { timeout: 30000 });
  terminatedProcess.kill('SIGTERM');
  await terminated;
  assert.equal(terminatedProcess.signalCode, null, 'SIGTERM must follow graceful application shutdown');
  assert.equal(terminatedProcess.exitCode, 0);
  application = undefined;
  assert.equal(await health(), null);
  assert.equal(alive(terminatedService), false);
  await eventually(() => readFile(probePids, 'utf8').then(text => text.trim().split('\n').map(Number).some(alive)), false);
  console.log('Relaunch after crash and SIGTERM: ownership reacquired, graceful exit0 retires service and shell probes');

  application = await launchOwner({ chromiumSandbox: true, executablePath, args: ['--background'], env, timeout: 60000 });
  await eventually(async () => (await health())?.state?._tag, 'Ready', 60000);
  const shutdownService = (await health()).pid;
  const shutdownProcess = application.process();
  const shutdownClosed = application.waitForEvent('close', { timeout: 30000 });
  await application.evaluate(({ powerMonitor }) => {
    powerMonitor.emit('shutdown', { preventDefault() { throw new Error('OS shutdown must not be vetoed'); } });
  });
  await shutdownClosed;
  assert.equal(shutdownProcess.exitCode, 0);
  application = undefined;
  assert.equal(await health(), null);
  assert.equal(alive(shutdownService), false);
  await eventually(() => readFile(probePids, 'utf8').then(text => text.trim().split('\n').map(Number).some(alive)), false);
  console.log('Simulated powerMonitor shutdown: no veto, graceful exit0 and owned service/probes retired; not a real OS logout test');

} finally {
  if (probeOwner && probeOwner.exitCode === null && probeOwner.signalCode === null) probeOwner.kill('SIGKILL');
  if (application) {
    const child = application.process();
    const timeout = setTimeout(() => child.kill('SIGKILL'), 5000);
    try { await application.close(); }
    finally { clearTimeout(timeout); }
  }
  if (incumbent) await new Promise(resolve => incumbent.close(resolve));
  await Promise.all([profile, failedProfile].map(path => rm(path, { recursive: true, force: true })));
}
