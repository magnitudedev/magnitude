const assert = require('node:assert/strict')
const { spawnSync } = require('node:child_process')
const native = require(process.argv[2])
assert.equal(process.platform, 'linux')
const probe = () => spawnSync('python3', ['-c', 'import fcntl; f=open("/var/lib/magnitude-desktop/installation.lock", "rb"); fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)'], { encoding: 'utf8' })
assert.equal(probe().status, 0, 'Fixture requires no running installed owner or installer')
const lease = native.acquireInstallationLease()
try {
  assert.notEqual(probe().status, 0, 'Retained shared lease excludes installation')
  const child = spawnSync(process.execPath, ['-e', `
    const fs = require('node:fs');
    for (const fd of fs.readdirSync('/proc/self/fd')) {
      let target; try { target = fs.readlinkSync('/proc/self/fd/' + fd) } catch { continue }
      if (target === '/var/lib/magnitude-desktop/installation.lock') process.exit(1);
    }
  `])
  assert.equal(child.status, 0, 'Service children do not inherit installation admission')
  assert.throws(() => native.releaseLock(lease), /Invalid ownership lock/)
  assert.throws(() => native.releaseInstallationLease({}), /Invalid installation admission/)
  assert.throws(() => native.releaseInstallationLease(Object.create(lease)), /Invalid installation admission/)
} finally { native.releaseInstallationLease(lease) }
native.releaseInstallationLease(lease)
assert.equal(probe().status, 0, 'Explicit release permits installation')
console.log('PASS shared installation admission, close-on-exec and exact capability release')
