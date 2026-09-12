import { _electron as electron } from 'playwright';
import assert from 'node:assert/strict';
import { mkdtemp, rm, readFile, writeFile, chmod } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawn, execFileSync } from 'node:child_process';
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
const alive = pid => {
  try { process.kill(pid, 0); return true; }
  catch (error) { if (error.code === 'ESRCH') return false; throw error; }
};
const eventually = async (read, expected, timeout = 10000) => {
  const deadline = Date.now() + timeout;
  let actual;
  do {
    actual = await read();
    if (JSON.stringify(actual) === JSON.stringify(expected)) return;
    await delay(100);
  } while (Date.now() < deadline);
  assert.deepEqual(actual, expected);
};
const invoke = args => new Promise((resolve, reject) => {
  const child = spawn(executablePath, args, { env, stdio: 'ignore' });
  const timeout = setTimeout(() => { child.kill('SIGKILL'); reject(new Error('Application contender did not exit')); }, 10000);
  child.once('error', error => { clearTimeout(timeout); reject(error); });
  child.once('exit', code => { clearTimeout(timeout); resolve(code); });
});
let application;
let probeOwner;
let legacy;
let legacyEngine;
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
    }, 10000);
    child.once('error', error => { clearTimeout(timeout); reject(error); });
    child.once('exit', code => { clearTimeout(timeout); resolve({ code, output }); });
  });
  assert.equal(rejected.code, 1, rejected.output);
  assert.match(rejected.output, /ApplicationOwnershipFailed|EEXIST/);
  assert.doesNotMatch(rejected.output, /Cause\.reduceWithContext|UnhandledPromiseRejection/);
  console.log('Rejected ownership path: original failure reported and background process exited');
  application = await electron.launch({ chromiumSandbox: true, executablePath, args: [], env: {
    ...env, MAGNITUDE_DEV_DATA_DIR: failedProfile,
    MAGNITUDE_ICN_PATH: join(failedProfile, 'absent-engine.json'),
  }, timeout: 30000 });
  const failedOwner = application.process();
  const failedWindow = await application.firstWindow();
  await failedWindow.waitForFunction(() => !!window.__magnitudeDesktop);
  const rejectedLogin = await failedWindow.evaluate(async () => {
    try { await window.__magnitudeDesktop.setLoginStartup(true); return null; }
    catch (error) { return { message: error.message }; }
  });
  assert.deepEqual(rejectedLogin, { message: 'Launch at login is available in the installed Magnitude app.' });
  console.log('Host action failure preserves actionable message across real contextBridge');
  await failedWindow.getByRole('button', { name: 'Status', exact: true }).click();
  for (let attempt = 0; attempt < 2; attempt++) {
    await failedWindow.getByText('Failed', { exact: true }).waitFor({ timeout: 20000 });
    await failedWindow.getByText(`The inference server binary was not found at ${join(failedProfile, 'bin/magnitude-inference')}`, { exact: true }).waitFor();
    await failedWindow.getByText(/^Registered with your desktop\./).waitFor();
    assert.equal(await failedWindow.getByText('Cleanup needs attention', { exact: false }).count(), 0);
    assert.equal(alive(failedOwner.pid), true);
    if (attempt === 0) {
      await failedWindow.getByRole('button', { name: 'Retry service', exact: true }).click();
      await failedWindow.getByText('Starting', { exact: true }).waitFor();
    }
  }
  const failedQuit = application.waitForEvent('close', { timeout: 10000 });
  await application.evaluate(({ Menu, BrowserWindow }) => {
    const window = BrowserWindow.getAllWindows()[0];
    const item = Menu.getApplicationMenu().items.flatMap(item => item.submenu?.items ?? []).find(item => item.label === 'Quit Magnitude');
    if (!item) throw new Error('Native full Quit action is missing');
    setImmediate(() => item.click({}, window, window.webContents));
  });
  await failedQuit;
  application = undefined;
  assert.equal(failedOwner.exitCode, 0);
  console.log('Missing inference engine: bounded startup failures retain safe detail and tray, Retry repeats cleanly, native Quit exits0');

  legacy = spawn(process.env.MAGNITUDE_TEST_BUN ?? 'bun', [fileURLToPath(new URL('./legacy-service.ts', import.meta.url)), profile], { detached: true, stdio: ['ignore', 'pipe', 'inherit'] });
  const legacyInfo = await new Promise((resolve, reject) => {
    let output = '';
    const timeout = setTimeout(() => reject(new Error('Legacy fixture did not become ready')), 10000);
    legacy.once('error', error => { clearTimeout(timeout); reject(error); });
    legacy.stdout.on('data', chunk => {
      output += chunk;
      if (!output.includes('\n')) return;
      clearTimeout(timeout);
      try { resolve(JSON.parse(output.trim())); } catch (error) { reject(error); }
    });
  });
  legacyEngine = legacyInfo.enginePid;
  await writeFile(join(profile, 'migration-preserved.txt'), 'preserve user data');
  application = await electron.launch({ chromiumSandbox: true, executablePath, args: ['--background'], env, timeout: 30000 });
  const app = application;
  const visibility = () => app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows().map(window => ({ visible: window.isVisible(), minimized: window.isMinimized() })));
  const health = () => fetch('http://127.0.0.1:11109/health').then(response => response.json()).catch(() => null);
  await eventually(visibility, [{ visible: false, minimized: false }], 2000);
  await eventually(() => readFile(probePids, 'utf8').then(text => text.trim().split('\n').map(Number).some(alive)).catch(() => false), true, 2000);
  console.log('Slow shell probe: owner and hidden window available while shell is still running');
  await eventually(async () => (await health())?.state?._tag, 'Ready', 60000);
  await eventually(visibility, [{ visible: false, minimized: false }]);
  const service = await health();
  if (process.env.MAGNITUDE_TEST_EXPECT_VERSION) assert.equal(service.version, process.env.MAGNITUDE_TEST_EXPECT_VERSION);
  if (process.env.MAGNITUDE_TEST_EXPECT_REVISION) assert.equal(service.revision, Number(process.env.MAGNITUDE_TEST_EXPECT_REVISION));
  if (process.env.MAGNITUDE_TEST_EXPECT_RPC_VERSION) assert.equal(service.rpcVersion, Number(process.env.MAGNITUDE_TEST_EXPECT_RPC_VERSION));
  assert.notEqual(service.pid, legacyInfo.pid);
  await eventually(() => Promise.resolve([legacyInfo.pid, legacyEngine].some(alive)), false);
  assert.equal(JSON.parse(await readFile(join(profile, 'desktop/legacy-migration.json'), 'utf8'))._tag, 'Complete');
  await assert.rejects(readFile(join(profile, 'acn/coordination.sqlite')), { code: 'ENOENT' });
  assert.equal(await readFile(join(profile, 'migration-preserved.txt'), 'utf8'), 'preserve user data');
  console.log('Legacy migration: old service and detached inference retired, checkpoint complete, user data preserved');
  console.log('Cold background launch: service Ready, window hidden');

  assert.deepEqual(await Promise.all(Array.from({ length: 4 }, () => invoke(['--background']))), [0, 0, 0, 0]);
  assert.equal((await health()).pid, service.pid);
  assert.deepEqual(await visibility(), [{ visible: false, minimized: false }]);
  console.log('Four concurrent background launches: same service, window hidden');

  assert.equal(await invoke([]), 0);
  await eventually(visibility, [{ visible: true, minimized: false }]);
  const window = await app.firstWindow();
  window.setDefaultTimeout(10000);
  await window.getByRole('button', { name: 'Skip setup', exact: true }).click();
  await window.getByRole('region', { name: 'Get started', exact: true }).waitFor({ state: 'hidden' });
  console.log('Fresh profile: explicit Skip completes onboarding without a model');
  await window.getByRole('button', { name: 'Connections', exact: true }).click();
  await window.getByRole('heading', { name: 'Connections', exact: true }).waitFor();
  await eventually(() => window.getByRole('button', { name: 'Connect', exact: true }).count(), 8);
  assert.equal(await window.getByRole('button', { name: 'Disconnect', exact: true }).count(), 0);
  assert.equal(await window.getByRole('button', { name: 'Connect', exact: true }).first().isDisabled(), true);
  const rejectedConnection = await window.evaluate(async () => {
    try { await window.__magnitudeDesktop.connect({ harness: 'codex' }); return null; }
    catch (error) { return error.message; }
  });
  assert.match(rejectedConnection, /No installed Magnitude models are available|Codex is not installed/);
  assert.doesNotMatch(rejectedConnection, /UnknownException|FiberFailure|Effect\.tryPromise|\n\s+at /);
  assert.equal(await window.getByRole('button', { name: 'Disconnect', exact: true }).count(), 0);
  console.log('Rejected connection preserves actionable host failure and does not create a managed connection');
  await window.getByRole('button', { name: 'Settings', exact: true }).click();
  const appVersion = await app.evaluate(({ app }) => app.getVersion());
  await window.getByText(`Version ${appVersion}`, { exact: true }).waitFor();
  const theme = window.getByRole('group', { name: 'Theme', exact: true });
  const backgrounds = [];
  for (const preference of ['light', 'dark']) {
    await theme.getByRole('button', { name: preference === 'light' ? 'Light' : 'Dark', exact: true }).click();
    await eventually(() => window.evaluate(() => document.documentElement.dataset.theme), preference);
    await eventually(() => app.evaluate(({ nativeTheme }) => nativeTheme.themeSource), preference);
    assert.equal(await window.evaluate(() => localStorage.getItem('magnitude.appearance')), preference);
    backgrounds.push(await window.locator('main').evaluate(element => getComputedStyle(element.parentElement).backgroundColor));
  }
  assert.notEqual(backgrounds[0], backgrounds[1], 'Light and dark must actually render different application backgrounds');
  const fonts = await window.evaluate(async () => {
    await document.fonts.ready;
    return {
      body: getComputedStyle(document.body).fontFamily,
      heading: getComputedStyle(document.querySelector('h1')).fontFamily,
      loaded: [...document.fonts].filter(font => font.status === 'loaded').map(font => font.family),
    };
  });
  assert.match(fonts.body, /Inter/);
  assert.match(fonts.heading, /Martian Mono/);
  assert.ok(fonts.loaded.some(family => family.includes('Inter')), 'Bundled Inter font must load');
  assert.ok(fonts.loaded.some(family => family.includes('Martian Mono')), 'Bundled Martian Mono font must load');
  await window.reload();
  await eventually(() => window.evaluate(() => document.documentElement.dataset.theme), 'dark');
  await window.getByRole('button', { name: 'Settings', exact: true }).click();
  assert.equal(await theme.getByRole('button', { name: 'Dark', exact: true }).getAttribute('aria-pressed'), 'true');
  await theme.getByRole('button', { name: 'System', exact: true }).click();
  await eventually(() => app.evaluate(({ nativeTheme }) => nativeTheme.themeSource), 'system');
  assert.equal(await window.evaluate(() => localStorage.getItem('magnitude.appearance')), null);
  await eventually(() => window.evaluate(() => document.documentElement.dataset.theme === (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')), true);
  console.log('Packaged appearance: Light/Dark/System, native theme, persisted reload, distinct backgrounds, and loaded Inter/Martian Mono fonts pass');
  await window.getByRole('button', { name: 'Status', exact: true }).click();
  await window.getByText('No downloads in progress.', { exact: true }).waitFor();
  await window.getByText(/^Registered with your desktop\./).waitFor();
  await window.getByRole('button', { name: 'Discover', exact: true }).click();
  await eventually(() => window.locator('main').evaluate(element => element.scrollHeight > element.clientHeight + 100), true);
  await window.locator('main').evaluate(element => { element.scrollTop = 100; });
  assert.ok(await window.locator('main').evaluate(element => element.scrollTop > 0));
  await window.getByRole('button', { name: 'Status', exact: true }).click();
  await window.getByRole('heading', { name: 'Status', exact: true }).waitFor();
  assert.equal(await window.locator('main').evaluate(element => element.scrollTop), 0);
  console.log('Page navigation: new destination starts at the top without inheriting catalog scroll');
  console.log('Fresh Connections: eight observed harnesses, no false ownership, connection requires an installed model');
  // Native macOS role invocation is covered by CUA with a separate passive visibility observer.
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
  const rendererReady = () => app.evaluate(async ({ BrowserWindow }) => {
    const contents = BrowserWindow.getAllWindows()[0].webContents;
    if (contents.isCrashed() || contents.isLoading()) return false;
    return Promise.race([
      contents.executeJavaScript('document.querySelector("h1")?.textContent === "Discover"'),
      new Promise(resolve => setTimeout(() => resolve(false), 1000)),
    ]);
  }).catch(() => false);
  await eventually(rendererReady, true);
  for (let attempt = 0; attempt < 3; attempt++) {
    console.log(`Crashing renderer: attempt ${attempt + 1}`);
    await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].webContents.forcefullyCrashRenderer());
    await eventually(rendererReady, true);
    console.log(`Renderer recovered: attempt ${attempt + 1}`);
    assert.equal((await health()).pid, service.pid);
    assert.deepEqual(await visibility(), [{ visible: false, minimized: false }]);
  }
  await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].webContents.forcefullyCrashRenderer());
  await delay(1000);
  assert.equal(await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].webContents.isCrashed()), true);
  assert.equal(await invoke(['--background']), 0);
  assert.equal(await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].webContents.isCrashed()), true);
  assert.equal((await health()).pid, service.pid);
  assert.equal(await invoke([]), 0);
  await eventually(rendererReady, true);
  assert.deepEqual(await visibility(), [{ visible: true, minimized: false }]);
  console.log('Renderer crashes: three bounded retries, background demand cannot renew them, explicit Open recovers the same service');

  const menuQuit = app.waitForEvent('close', { timeout: 10000 });
  await app.evaluate(({ BrowserWindow, Menu }) => {
    const window = BrowserWindow.getAllWindows()[0];
    const item = Menu.getApplicationMenu().items.flatMap(item => item.submenu?.items ?? []).find(item => item.label === 'Quit Magnitude');
    if (!item) throw new Error('Native full Quit action is missing');
    setImmediate(() => item.click({}, window, window.webContents));
  });
  await menuQuit;
  application = undefined;
  assert.equal(await health(), null);
  assert.throws(() => process.kill(service.pid, 0));
  await eventually(() => readFile(probePids, 'utf8').then(text => text.trim().split('\n').map(Number).some(alive)), false);
  console.log('Native menu Quit: service and shell probe groups absent, endpoint closed');

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

  application = await electron.launch({ chromiumSandbox: true, executablePath, args: ['--background'], env, timeout: 30000 });
  await eventually(async () => (await health())?.state?._tag, 'Ready', 60000);
  const reopened = await application.firstWindow();
  await reopened.getByRole('button', { name: 'Download', exact: true }).first().waitFor();
  assert.equal(await reopened.getByRole('button', { name: 'Skip setup', exact: true }).count(), 0);
  assert.deepEqual(await application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows().map(window => window.isVisible())), [false]);
  console.log('Completed onboarding persists after full Quit and hidden relaunch');
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

  application = await electron.launch({ chromiumSandbox: true, executablePath, args: ['--background'], env, timeout: 30000 });
  await eventually(async () => (await health())?.state?._tag, 'Ready', 60000);
  assert.notEqual((await health()).pid, replacement.pid);
  const terminatedService = (await health()).pid;
  const terminatedProcess = application.process();
  const terminated = application.waitForEvent('close', { timeout: 15000 });
  terminatedProcess.kill('SIGTERM');
  await terminated;
  assert.equal(terminatedProcess.signalCode, null, 'SIGTERM must follow graceful application shutdown');
  assert.equal(terminatedProcess.exitCode, 0);
  application = undefined;
  assert.equal(await health(), null);
  assert.equal(alive(terminatedService), false);
  await eventually(() => readFile(probePids, 'utf8').then(text => text.trim().split('\n').map(Number).some(alive)), false);
  console.log('Relaunch after crash and SIGTERM: ownership reacquired, graceful exit0 retires service and shell probes');

  application = await electron.launch({ chromiumSandbox: true, executablePath, args: ['--background'], env, timeout: 30000 });
  await eventually(async () => (await health())?.state?._tag, 'Ready', 60000);
  const shutdownService = (await health()).pid;
  const shutdownProcess = application.process();
  const shutdownClosed = application.waitForEvent('close', { timeout: 15000 });
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
  for (const pid of [legacyEngine, legacy?.pid]) if (pid && alive(pid)) {
    try { process.kill(-pid, 'SIGKILL'); } catch (error) { if (error.code !== 'ESRCH') throw error; }
  }
  await Promise.all([profile, failedProfile].map(path => rm(path, { recursive: true, force: true })));
}
