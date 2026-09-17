import { chromium } from 'playwright'
import assert from 'node:assert/strict'

const browser = await chromium.launch({ headless: true })
try {
  for (const reducedMotion of ['no-preference', 'reduce']) {
    const page = await browser.newPage({ viewport: { width: 1600, height: 1200 }, reducedMotion })
    await page.goto('http://127.0.0.1:6091/loading.html')
    await page.getByRole('navigation').waitFor()
    await page.evaluate(() => window.loadingFixture.setPhase('preference'))
    const panel = page.getByLabel('Top recommendations')
    const rows = page.getByLabel('Recommended models', { exact: true }).getByRole('button')
    const path = panel.locator('svg[role="img"] path')
    await path.waitFor()
    const panelNode = await panel.elementHandle()
    const pathNode = await path.elementHandle()
    await page.getByRole('button', { name: 'Fastest', exact: true }).click()
    await page.waitForTimeout(350)
    const fastest = await rows.first().innerText()
    const fastShape = await path.evaluate(el => getComputedStyle(el).d)
    await rows.nth(3).click()
    await page.getByRole('button', { name: 'Smartest', exact: true }).click()
    assert.notEqual(await rows.first().innerText(), fastest, 'Preference must reorder models')
    assert.equal(await rows.first().getAttribute('aria-pressed'), 'true', 'Preference must reset selection to best match')
    assert.ok(await panelNode.evaluate(el => el.isConnected), 'Panel must not remount')
    assert.ok(await pathNode.evaluate(el => el.isConnected), 'Radar must not remount')
    assert.equal(await panel.evaluate(el => getComputedStyle(el).opacity), '1', 'Panel must not fade out')
    assert.equal(await panel.evaluate(el => getComputedStyle(el).animationName), 'none')
    await page.waitForTimeout(50)
    const moving = await path.evaluate(el => getComputedStyle(el).d)
    await page.waitForTimeout(350)
    const settled = await path.evaluate(el => getComputedStyle(el).d)
    assert.notEqual(settled, fastShape, 'Radar must reflect the selected model')
    if (reducedMotion === 'reduce') assert.equal(moving, settled, 'Reduced motion must update immediately')
    else assert.notEqual(moving, settled, 'Radar must interpolate between profiles')
    // Returning to an earlier preference must not resurrect its old manual selection.
    await page.getByRole('button', { name: 'Fastest', exact: true }).click()
    assert.equal(await rows.first().getAttribute('aria-pressed'), 'true')
    await page.close()
  }
  console.log('PASS preference updates preserve panel and radar nodes, select the best match, and interpolate geometry while respecting reduced motion')
} finally {
  await browser.close()
}
