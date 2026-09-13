import assert from 'node:assert/strict'
import { createHash, createPublicKey, verify } from 'node:crypto'
import { execFileSync } from 'node:child_process'
import { once } from 'node:events'
import { mkdir, readFile, writeFile, access, unlink } from 'node:fs/promises'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { setTimeout as delay } from 'node:timers/promises'
import { _electron as electron } from 'playwright'

assert.equal(process.platform, 'win32')
assert.equal(process.versions.bun, undefined)
const from = process.env.MAGNITUDE_WINDOWS_ACCEPTANCE_FROM
const to = process.env.MAGNITUDE_WINDOWS_ACCEPTANCE_TO
assert.match(from ?? '', /^0\.0\.\d+$/)
assert.match(to ?? '', /^0\.0\.\d+$/)
const root = process.env.MAGNITUDE_WINDOWS_ACCEPTANCE_ROOT
assert.ok(root)
const installation = join(process.env.LOCALAPPDATA, 'Programs', 'Magnitude')
await assert.rejects(access(installation), 'Consumer requires a fresh disposable Windows runner')
const evidence = join(root, 'consumer')
await mkdir(evidence)
const envelopes = JSON.parse(await readFile(join(root, from, 'artifacts/prepared-manifests.json'), 'utf8'))
assert.equal(envelopes.length, 1)
const envelope = envelopes[0]
const json = Buffer.from(envelope.payload, 'base64').toString()
assert.ok(verify(null, Buffer.from('magnitude-release-v1\n' + json),
  createPublicKey(await readFile(new URL('../../../packages/release/resources/distribution/acceptance.pub.pem', import.meta.url))),
  Buffer.from(envelope.signature, 'base64')))
const manifest = JSON.parse(json)
assert.equal(manifest.version, from)
assert.equal(manifest.artifact.target.package, 'windows-exe')
const installer = join(evidence, 'downloaded-installer.exe')
execFileSync('bun', [fileURLToPath(new URL('./windows-download-installer.ts', import.meta.url))], { stdio: 'inherit', timeout: 15 * 60000 })
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
  const preferencesPath = join(data, 'updates/preferences.json')
  for (const enabled of [false, true, false]) {
    await automatic.click()
    await automatic.and(page.locator(enabled ? ':checked' : ':not(:checked)')).waitFor({ timeout: 10000 })
    assert.equal(JSON.parse(await readFile(preferencesPath, 'utf8')).autoDownload, enabled)
  }
  const keyPath = join(data, 'updates/installation-key.pem')
  const identity = await readFile(keyPath)
  const publicBytes = createPublicKey(identity).export({ type: 'spki', format: 'der' }).subarray(-32)
  const installationId = createHash('sha256').update(publicBytes).digest('hex')
  await writeFile(join(evidence, 'installation.json'), JSON.stringify({ installationId, version: from, at: new Date().toISOString() }, null, 2))
  console.log('WAITING for acceptance channel', to, 'installation', installationId)
  const deadline = Date.now() + 15 * 60000
  while (!(await page.getByRole('button', { name: 'Download update', exact: true }).isVisible())) {
    assert.ok(Date.now() < deadline, 'Acceptance channel was not promoted before the consumer deadline')
    await page.getByRole('button', { name: 'Check for updates', exact: true }).click()
    await delay(30000)
  }
  await page.screenshot({ path: join(evidence, 'available.png'), fullPage: true })
  assert.ok(cli(['update', 'status']).includes(`${to} is available`))
  await page.getByRole('button', { name: 'Download update', exact: true }).click()
  await page.getByRole('button', { name: 'Restart to update', exact: true }).waitFor({ timeout: 600000 })
  await page.screenshot({ path: join(evidence, 'ready.png'), fullPage: true })
  assert.ok(cli(['update', 'status']).includes(`${to} is ready to install`))
  console.log('PASS application downloaded and verified the offered installer')
  const original = app.process()
  const exited = once(original, 'exit', { signal: AbortSignal.timeout(60000) })
  await page.getByRole('button', { name: 'Restart to update', exact: true }).click({ noWaitAfter: true, timeout: 10000 }).catch(error => {
    console.log('Restart closed the automation connection:', error.message)
  })
  await exited
  console.log('PASS original application process exited for replacement', original.pid)
  app = undefined
  const restarted = Date.now() + 120000
  let status = ''
  while (Date.now() < restarted) {
    try {
      if (cli(['--version']).trim() === to) {
        status = cli(['service', 'status'])
        if (/Tray\s+Registered/i.test(status)) break
      }
    } catch {}
    await delay(1000)
  }
  assert.match(status, /Tray\s+Registered/i, 'Updated app must relaunch with its native tray')
  assert.equal(cli(['--version']).trim(), to)
  assert.deepEqual(await readFile(keyPath), identity, 'Update changed installation identity')
  assert.equal(JSON.parse(await readFile(preferencesPath, 'utf8')).autoDownload, false)
  await writeFile(join(evidence, 'after-relaunch.txt'), status)
  console.log('PASS real Settings download/restart, updated installed version, automatic owner/tray relaunch and identity preservation')
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
  await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows().forEach(window => window.close()))
  assert.equal(await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows().some(window => window.isVisible())), false)
  assert.match(cli(['service', 'status']), /Tray\s+Registered/i)
  execFileSync('powershell', ['-NoProfile', '-File', fileURLToPath(new URL('./windows-tray-acceptance.ps1', import.meta.url)), '-Evidence', evidence, '-MenuAction', 'Open Magnitude'], { stdio: 'inherit', timeout: 60000 })
  await page.getByRole('heading', { name: 'Discover', exact: true }).waitFor({ timeout: 10000 })
  assert.equal(await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows().some(window => window.isVisible())), true)
  await writeFile(join(evidence, 'accepted.json'), JSON.stringify({ installationId, from, to, at: new Date().toISOString(), status }, null, 2))
  console.log('PASS installed updated Settings, persisted auto-download preference and real up-to-date check')
} catch (error) {
  console.error(error)
  await writeFile(join(evidence, 'failure.txt'), String(error.stack ?? error))
  throw error
} finally {
  if (app && app.process().exitCode === null) {
    const visible = app.windows()[0]
    if (visible) await visible.screenshot({ path: join(evidence, 'last-window.png'), fullPage: true, timeout: 5000 }).catch(() => {})
    await Promise.race([app.close(), delay(10000).then(() => {
      if (app.process().exitCode === null) app.process().kill()
    })])
  }
}
