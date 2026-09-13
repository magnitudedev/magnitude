import assert from 'node:assert/strict'
import { createHash, createPublicKey, verify } from 'node:crypto'
import { execFileSync } from 'node:child_process'
import { mkdir, readFile, writeFile, access, unlink } from 'node:fs/promises'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { setTimeout as delay } from 'node:timers/promises'
import { _electron as electron } from 'playwright'

assert.equal(process.platform, 'win32')
assert.equal(process.versions.bun, undefined)
const root = process.env.MAGNITUDE_WINDOWS_ACCEPTANCE_ROOT
assert.ok(root)
const installation = join(process.env.LOCALAPPDATA, 'Programs', 'Magnitude')
await assert.rejects(access(installation), 'Consumer requires a fresh disposable Windows runner')
const evidence = join(root, 'consumer')
await mkdir(evidence)
const envelopes = JSON.parse(await readFile(join(root, '0.0.30/artifacts/prepared-manifests.json'), 'utf8'))
assert.equal(envelopes.length, 1)
const envelope = envelopes[0]
const json = Buffer.from(envelope.payload, 'base64').toString()
assert.ok(verify(null, Buffer.from('magnitude-release-v1\n' + json),
  createPublicKey(await readFile(new URL('../../../packages/release/resources/distribution/acceptance.pub.pem', import.meta.url))),
  Buffer.from(envelope.signature, 'base64')))
const manifest = JSON.parse(json)
assert.equal(manifest.version, '0.0.30')
assert.equal(manifest.artifact.target.package, 'windows-exe')
assert.match(manifest.artifact.path, /^releases\/0\.0\.30\/acceptance-[a-f0-9]{40}-[a-zA-Z0-9.-]+\.exe$/)
const response = await fetch('https://5r3lqtpag4uzvtxd.public.blob.vercel-storage.com/' + manifest.artifact.path, { signal: AbortSignal.timeout(180000) })
assert.equal(response.status, 200)
const installerBytes = Buffer.from(await response.arrayBuffer())
assert.equal(installerBytes.length, manifest.artifact.bytes)
assert.equal(createHash('sha256').update(installerBytes).digest('hex'), manifest.artifact.sha256)
const installer = join(evidence, 'downloaded-installer.exe')
await writeFile(installer, installerBytes)
execFileSync('pwsh', ['-NoProfile', '-Command', '& { param($p) $s=Get-AuthenticodeSignature -LiteralPath $p; if ($s.Status -ne "Valid") { throw "Invalid downloaded installer signature" }; $i=Start-Process -FilePath $p -ArgumentList /S -PassThru -Wait; if ($i.ExitCode -ne 0) { throw "Installation failed" } }', installer], { stdio: 'inherit', timeout: 120000 })
await unlink(installer)
console.log('PASS actual hosted installer download, publisher signature and native fresh install')
const data = join(root, 'consumer-profile')
const state = join(data, 'desktop')
const env = { ...process.env, MAGNITUDE_DEV_DATA_DIR: data, MAGNITUDE_DESKTOP_STATE_DIR: state, MAGNITUDE_DEV_PORT: '11143' }
const executablePath = join(installation, 'Magnitude.exe')
const cli = args => execFileSync(join(installation, 'resources/magnitude.exe'), args, { env, encoding: 'utf8', timeout: 30000 })
let app
try {
  app = await electron.launch({ executablePath, env, timeout: 30000 })
  let page = await app.firstWindow()
  await page.getByRole('button', { name: 'Settings', exact: true }).click()
  const automatic = page.getByRole('checkbox', { name: 'Auto-download updates' })
  await automatic.uncheck()
  const keyPath = join(data, 'updates/installation-key.pem')
  const identity = await readFile(keyPath)
  const publicBytes = createPublicKey(identity).export({ type: 'spki', format: 'der' }).subarray(-32)
  const installationId = createHash('sha256').update(publicBytes).digest('hex')
  await writeFile(join(evidence, 'installation.json'), JSON.stringify({ installationId, version: '0.0.30', at: new Date().toISOString() }, null, 2))
  console.log('WAITING for acceptance channel 0.0.31; installation', installationId)
  const deadline = Date.now() + 15 * 60000
  while (!(await page.getByRole('button', { name: 'Download update', exact: true }).isVisible())) {
    assert.ok(Date.now() < deadline, 'Acceptance channel was not promoted before the consumer deadline')
    await page.getByRole('button', { name: 'Check for updates', exact: true }).click()
    await delay(30000)
  }
  await page.screenshot({ path: join(evidence, 'available.png'), fullPage: true })
  assert.match(cli(['update', 'status']), /0\.0\.31 is available/)
  await page.getByRole('button', { name: 'Download update', exact: true }).click()
  await page.getByRole('button', { name: 'Restart to update', exact: true }).waitFor({ timeout: 180000 })
  await page.screenshot({ path: join(evidence, 'ready.png'), fullPage: true })
  assert.match(cli(['update', 'status']), /0\.0\.31 is ready to install/)
  const closed = app.waitForEvent('close', { timeout: 60000 })
  await page.getByRole('button', { name: 'Restart to update', exact: true }).click()
  await closed
  app = undefined
  const restarted = Date.now() + 120000
  let status = ''
  while (Date.now() < restarted) {
    try {
      if (cli(['--version']).trim() === '0.0.31') {
        status = cli(['service', 'status'])
        if (/Tray\s+Registered/i.test(status)) break
      }
    } catch {}
    await delay(1000)
  }
  assert.match(status, /Tray\s+Registered/i, 'Updated app must relaunch with its native tray')
  assert.equal(cli(['--version']).trim(), '0.0.31')
  assert.deepEqual(await readFile(keyPath), identity, 'Update changed installation identity')
  assert.equal(JSON.parse(await readFile(join(data, 'updates/preferences.json'), 'utf8')).autoDownload, false)
  await writeFile(join(evidence, 'after-relaunch.txt'), status)
  console.log('PASS real Settings download/restart, installed version 0.0.31, automatic owner/tray relaunch and identity preservation')
  cli(['service', 'stop'])
  await delay(1000)
  app = await electron.launch({ executablePath, env, timeout: 30000 })
  page = await app.firstWindow()
  await page.getByRole('button', { name: 'Settings', exact: true }).click()
  assert.equal(await page.getByRole('checkbox', { name: 'Auto-download updates' }).isChecked(), false)
  await page.getByRole('button', { name: 'Check for updates', exact: true }).click()
  await page.getByText('You’re up to date.', { exact: true }).waitFor({ timeout: 30000 })
  await page.screenshot({ path: join(evidence, 'updated-settings.png'), fullPage: true })
  execFileSync('powershell', ['-NoProfile', '-File', fileURLToPath(new URL('./windows-tray-acceptance.ps1', import.meta.url)), '-Evidence', evidence], { stdio: 'inherit', timeout: 60000 })
  await page.getByRole('heading', { name: 'Discover', exact: true }).waitFor({ timeout: 10000 })
  await writeFile(join(evidence, 'accepted.json'), JSON.stringify({ installationId, from: '0.0.30', to: '0.0.31', at: new Date().toISOString(), status }, null, 2))
  console.log('PASS installed updated Settings, persisted auto-download preference and real up-to-date check')
} finally {
  if (app) {
    const visible = app.windows()[0]
    if (visible) await visible.screenshot({ path: join(evidence, 'last-window.png'), fullPage: true }).catch(() => {})
    await app.close()
  }
}
