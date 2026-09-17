import assert from 'node:assert/strict'
import { mkdir, mkdtemp, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { execFileSync } from 'node:child_process'
import { pathToFileURL } from 'node:url'
const { _electron } = await import(pathToFileURL(process.env.PLAYWRIGHT_MODULE).href)
const output = process.env.CHROME_EVIDENCE
await mkdir(output, { recursive: true })
const profile = await mkdtemp(join(tmpdir(), 'magnitude-chrome-'))
const app = await _electron.launch({ executablePath: process.env.MAGNITUDE_TEST_APP, env: { ...process.env, MAGNITUDE_DEV_DATA_DIR: profile, MAGNITUDE_DEV_PORT: '11279', MAGNITUDE_SHELL_ENV_INHERITED: '1' }, timeout: 60000 })
const evidence = { platform: process.platform, checks: [] }
const wait = ms => new Promise(resolve => setTimeout(resolve, ms))
try {
  const page = await app.firstWindow()
  await page.getByRole('button', { name: 'Usage', exact: true }).waitFor({ timeout: 60000 })
  await page.getByRole('button', { name: 'Usage', exact: true }).click()
  await page.getByRole('heading', { name: 'Usage', exact: true }).waitFor()
  evidence.checks.push('navigation remains clickable')
  const integrated = process.platform !== 'linux'
  assert.equal(await page.locator('[data-window-drag-region]').count() > 0, integrated)
  const expandedContentWidth = (await page.locator('[data-page-content]').boundingBox()).width
  await page.getByRole('button', {name:'Collapse sidebar',exact:true}).click()
  await wait(300)
  await page.getByRole('button', {name:'Expand sidebar',exact:true}).waitFor()
  assert.equal((await page.locator('aside').boundingBox()).width, 0)
  assert.equal((await page.locator('[data-page-content]').boundingBox()).width, expandedContentWidth)
  await page.getByRole('button', {name:'Expand sidebar',exact:true}).click()
  await wait(300)
  assert.equal((await page.locator('aside').boundingBox()).width, 224)
  evidence.checks.push('sidebar fully hides and expands')
  evidence.geometry = await app.evaluate(({BrowserWindow}) => {const w=BrowserWindow.getAllWindows()[0];return {bounds:w.getBounds(),content:w.getContentBounds(),minimum:w.getMinimumSize()}})
  if (process.platform === 'darwin') {
    evidence.trafficLights = await app.evaluate(({BrowserWindow}) => BrowserWindow.getAllWindows()[0].getWindowButtonPosition())
    assert.deepEqual(evidence.trafficLights,{x:16,y:14})
  }
  if (process.platform === 'win32') {
    evidence.overlay = await page.evaluate(() => {const r=navigator.windowControlsOverlay.getTitlebarAreaRect();return {visible:navigator.windowControlsOverlay.visible,x:r.x,width:r.width,height:r.height,viewport:innerWidth}})
    assert.equal(evidence.overlay.visible,true)
    assert.ok(evidence.overlay.width < evidence.overlay.viewport)
  }
  if (process.platform === 'win32' && process.env.NATIVE_CHROME_SCRIPT) {
    const handle = await app.evaluate(({BrowserWindow}) => BrowserWindow.getAllWindows()[0].getNativeWindowHandle().readBigUInt64LE().toString())
    evidence.nativeHitTargets = JSON.parse(execFileSync('powershell.exe', ['-NoProfile','-ExecutionPolicy','Bypass','-File',process.env.NATIVE_CHROME_SCRIPT,handle,'400',String(Math.round(evidence.overlay.width + (evidence.overlay.viewport-evidence.overlay.width)/2))], {encoding:'utf8'}))
    assert.equal(evidence.nativeHitTargets.caption,2)
    assert.equal(evidence.nativeHitTargets.maximize,9)
    evidence.checks.push('native caption drag target and maximize hit target for Snap Layouts')
  }
  if (process.platform === 'linux') {
    const handle = await app.evaluate(({BrowserWindow}) => BrowserWindow.getAllWindows()[0].getNativeWindowHandle().readUInt32LE().toString())
    evidence.nativeFrame = execFileSync('xprop',['-id',handle,'_NET_FRAME_EXTENTS'],{encoding:'utf8'}).trim()
    const frame = evidence.nativeFrame.split('=')[1].split(',').map(Number)
    assert.ok(frame[2] > 0)
    evidence.checks.push('native Linux window-manager decorations')
  }
  for(const theme of ['light','dark']) {
    await app.evaluate(({nativeTheme}, theme) => {nativeTheme.themeSource=theme}, theme)
    await page.evaluate(theme => { document.documentElement.dataset.theme=theme }, theme)
    await wait(350)
    await page.screenshot({path:join(output, `${theme}.png`)})
    if (process.platform !== 'darwin') {
      const nativeImage = await app.evaluate(async ({desktopCapturer}) => {
        const sources=await desktopCapturer.getSources({types:['window'],thumbnailSize:{width:1280,height:900}})
        return sources.find(source=>source.name==='Magnitude')?.thumbnail.toPNG().toString('base64')
      })
      if (nativeImage) await writeFile(join(output,`${theme}-native.png`),Buffer.from(nativeImage,'base64'))
    }
    evidence.checks.push(`${theme} appearance`)
  }
  await app.evaluate(({BrowserWindow}) => BrowserWindow.getAllWindows()[0].minimize())
  await wait(500)
  assert.equal(await app.evaluate(({BrowserWindow}) => BrowserWindow.getAllWindows()[0].isMinimized()),true)
  await app.evaluate(({BrowserWindow}) => BrowserWindow.getAllWindows()[0].restore())
  await wait(500)
  evidence.checks.push('minimize and restore')
  await app.evaluate(({BrowserWindow}) => BrowserWindow.getAllWindows()[0].maximize())
  await wait(700)
  assert.equal(await app.evaluate(({BrowserWindow}) => BrowserWindow.getAllWindows()[0].isMaximized()),true)
  await app.evaluate(({BrowserWindow}) => BrowserWindow.getAllWindows()[0].unmaximize())
  await wait(500)
  evidence.checks.push('maximize and restore')
  await app.evaluate(({BrowserWindow}) => BrowserWindow.getAllWindows()[0].setFullScreen(true))
  await wait(1800)
  assert.equal(await app.evaluate(({BrowserWindow}) => BrowserWindow.getAllWindows()[0].isFullScreen()),true)
  await app.evaluate(({BrowserWindow}) => BrowserWindow.getAllWindows()[0].setFullScreen(false))
  await wait(1800)
  evidence.checks.push('fullscreen and exit')
  await app.evaluate(({BrowserWindow}) => BrowserWindow.getAllWindows()[0].setSize(800,600))
  await wait(500)
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth > innerWidth),false)
  await page.screenshot({path:join(output,'minimum-size.png')})
  evidence.checks.push('minimum size without page overflow')
  await app.evaluate(({BrowserWindow}) => BrowserWindow.getAllWindows()[0].close())
  await wait(500)
  assert.equal(await app.evaluate(({BrowserWindow}) => BrowserWindow.getAllWindows()[0].isVisible()),false)
  await app.evaluate(({BrowserWindow}) => BrowserWindow.getAllWindows()[0].show())
  evidence.checks.push('close hides retained window; reopen works')
  await writeFile(join(output,'result.json'),JSON.stringify(evidence,null,2))
  console.log(JSON.stringify(evidence,null,2))
  if (process.env.CHROME_MANUAL_CHECK) await wait(60000)
} finally { await app.close() }
