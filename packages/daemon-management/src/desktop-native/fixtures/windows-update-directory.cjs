const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const { execFileSync } = require('node:child_process')
const native = require(process.argv[2])
assert.equal(process.platform, 'win32')
const scratch = fs.mkdtempSync(path.join(os.tmpdir(), 'magnitude-update-directory-'))
const root = path.join(scratch, 'profile')
native.preparePrivateDirectory(root)
const updates = path.join(root, 'updates')
const target = path.join(root, 'unrelated')
try {
  fs.mkdirSync(target)
  fs.writeFileSync(path.join(target, 'keep'), 'preserved')
  fs.symlinkSync(target, updates, 'junction')
  assert.throws(() => native.recoverUpdateDirectory(updates))
  assert.equal(fs.readFileSync(path.join(target, 'keep'), 'utf8'), 'preserved')
  fs.unlinkSync(updates)
  fs.mkdirSync(updates, { mode: 0o700 })
  // Keep inherited permissions, but make ownership independent of elevated-token defaults.
  const ownerScript = "$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; $p=$env:MAGNITUDE_TEST_CACHE; $a=Get-Acl -LiteralPath $p; " +
    "$a.SetOwner([Security.Principal.WindowsIdentity]::GetCurrent().User); Set-Acl -LiteralPath $p -AclObject $a"
  execFileSync('powershell.exe', ['-NoProfile', '-NonInteractive', '-EncodedCommand', Buffer.from(ownerScript, 'utf16le').toString('base64')],
    { env: { ...process.env, MAGNITUDE_TEST_CACHE: updates } })
  const child = path.join(updates, 'desktop-update-abc123')
  fs.symlinkSync(target, child, 'junction')
  assert.throws(() => native.recoverUpdateDirectory(updates))
  assert.equal(fs.readFileSync(path.join(target, 'keep'), 'utf8'), 'preserved')
  fs.unlinkSync(child)
  assert.equal(native.recoverUpdateDirectory(updates), true)
  assert.equal(native.recoverUpdateDirectory(updates), false)
  assert.equal(fs.readdirSync(root).filter(name => name.startsWith('updates-retired-')).length, 1)
  console.log('PASS update directory recovery rejects root and child junctions and preserves unrelated files')
} finally {
  fs.rmSync(scratch, { recursive: true, force: true })
}
