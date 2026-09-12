// Copied beside the freshly built addon before Bun compilation; this require embeds it.
const native = require('./desktop-host.node');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
assert.equal(process.platform, 'win32', 'Requires native Windows execution');
assert.ok(require('bun').embeddedFiles.some(file => file.name.includes('desktop-host')), 'Addon must be embedded');
const root = fs.mkdtempSync(path.join(require('node:os').tmpdir(), 'magnitude-embedded-native-'));
let lock;
let observed;
try {
  const folder = native.localAppDataDirectory();
  assert.ok(path.win32.isAbsolute(folder));
  process.env.LOCALAPPDATA = '\\\\invalid\\environment-profile';
  assert.equal(native.localAppDataDirectory(), folder);
  const directory = path.join(root, 'private');
  assert.equal(native.inspectApplicationEndpoint(directory), null);
  lock = native.acquireLock(path.join(directory, 'application.lock'));
  assert.ok(lock);
  assert.equal(native.inspectApplicationEndpoint(directory), native.lockEndpoint(lock));
  observed = native.observeProcess(process.pid);
  assert.ok(observed);
  assert.equal(native.observedProcessExited(observed), false);
  assert.equal(native.observedProcessDetails(observed).pid, process.pid);
  console.log('PASS embedded Windows native known-folder, ownership and process observation');
} finally {
  if (observed) native.releaseObservedProcess(observed);
  if (lock) native.releaseLock(lock);
  fs.rmSync(root, { recursive: true, force: true });
}
