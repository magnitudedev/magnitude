import { chromium } from 'playwright'
import assert from 'node:assert/strict'
import { mkdir } from 'node:fs/promises'

const output = new URL('../../../specs/26-09-15/page-skeletons/screenshots/', import.meta.url).pathname
await mkdir(output, { recursive: true })
const browser = await chromium.launch({ headless: true })
const errors = []
try {
  for (const theme of ['light', 'dark']) for (const width of [800, 1120, 1600]) {
    const page = await browser.newPage({ viewport: { width, height: 1800 }, reducedMotion: 'reduce' })
    page.on('pageerror', error => errors.push(error.message))
    await page.goto('http://127.0.0.1:6091/loading.html')
    await page.getByRole('navigation').waitFor()
    await page.evaluate(theme => {
      document.documentElement.dataset.theme = theme
      window.loadingFixture.setPhase('loaded')
    }, theme)
    const panel = page.getByLabel('Top recommendations')
    const profile = page.getByLabel('Selected model profile')
    const chart = profile.getByRole('img', { name: /capability profile/ })
    const button = profile.getByRole('button', { name: /^Download/ })
    await button.waitFor()
    const before = { panel: await panel.boundingBox(), profile: await profile.boundingBox() }
    for (const phase of ['download-started', 'download-complete', 'download-unknown', 'download-verifying']) {
      await page.evaluate(phase => window.loadingFixture.setPhase(phase), phase)
      const progress = profile.getByRole('progressbar')
      await progress.waitFor()
      assert.deepEqual(await panel.boundingBox(), before.panel, `${theme} ${width} ${phase}: panel shifted`)
      assert.deepEqual(await profile.boundingBox(), before.profile, `${theme} ${width} ${phase}: profile shifted`)
      assert.equal(await chart.isVisible(), false, 'Chart must be replaced during download')
      assert.equal(await profile.getByRole('button', { name: 'Profile', exact: true }).count(), 0)
      assert.ok(await profile.getByText('Download speed', {exact:true}).isVisible())
      assert.ok(await profile.getByText('Time remaining', {exact:true}).isVisible())
      if (phase === 'download-started') {
        assert.ok(await profile.getByText('295 MB / 2.3 GB', {exact:true}).isVisible())
        assert.ok(await profile.getByText('13 MB/s', {exact:true}).isVisible())
        assert.ok(await profile.getByText('About 3 min', {exact:true}).isVisible())
      }
      assert.ok(await profile.getByRole('button', { name: 'Cancel download', exact: true }).isEnabled())
      const measured = phase === 'download-started' || phase === 'download-complete' || phase === 'download-verifying'
      const value = await progress.getAttribute('aria-valuenow')
      if (measured) {
        assert.ok(Math.abs(Number(value) - (phase === 'download-complete' ? 100 : 295 / 2300 * 100)) < 0.01)
        assert.match(await progress.getAttribute('aria-valuetext'), /\//)
      } else assert.equal(value, null, 'Unknown progress must remain indeterminate')
      assert.ok(await page.locator('main').evaluate(el => el.scrollWidth <= el.clientWidth), 'Horizontal overflow')
      if (phase === 'download-started') await panel.screenshot({ path: `${output}download-${theme}-${width}.png` })
    }
    await page.evaluate(() => window.loadingFixture.setPhase('loaded'))
    await chart.waitFor()
    assert.deepEqual(await panel.boundingBox(), before.panel, 'Returning to profile shifted panel')
    await page.close()
  }
  assert.deepEqual(errors, [])
  console.log('PASS download progress: stable panel geometry and profile restoration; measured and indeterminate states; cancel access; three widths and both themes')
} finally {
  await browser.close()
}
