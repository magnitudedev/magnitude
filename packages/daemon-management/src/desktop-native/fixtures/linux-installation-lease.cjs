const assert = require('node:assert/strict')
const fs = require('node:fs')
const { spawnSync } = require('node:child_process')
const addon = require(process.argv[2])
if (process.argv[3] === 'invalid') {
  assert.throws(() => addon.adoptInstallationLease())
} else {
  const path = '/var/lib/magnitude-desktop/installation.lock'
  const flags = () => parseInt(fs.readFileSync('/proc/self/fdinfo/9', 'utf8').match(/^flags:\s+([0-7]+)$/m)[1], 8)
  assert.equal(fs.readlinkSync('/proc/self/fd/9'), path)
  addon.adoptInstallationLease()
  assert.notEqual(flags() & 0o2000000, 0)
  assert.equal(fs.readlinkSync('/proc/self/fd/9'), path)
  const blocked = spawnSync('flock', ['--exclusive', '--nonblock', path, 'true'])
  assert.equal(blocked.status, 1, 'the desktop still excludes package replacement')
  const child = spawnSync('/bin/sh', ['-c', 'for f in /proc/self/fd/*; do readlink "$f"; done'], { encoding: 'utf8' })
  assert.ok(!child.stdout.includes(path), 'children must not retain installation admission')
}
console.log('PASS Linux installation lease admission and descriptor ownership')
