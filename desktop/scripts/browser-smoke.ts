import { strict as assert } from "node:assert"
import { spawn, type ChildProcess } from "node:child_process"
import { access, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises"
import { createServer } from "node:http"
import type { AddressInfo } from "node:net"
import { createRequire } from "node:module"
import { tmpdir } from "node:os"
import { dirname, join } from "node:path"
import { fileURLToPath } from "node:url"
import { chromium, type Browser, type BrowserContext, type Page } from "playwright"

const fixture = createServer((request, response) => {
    const url = new URL(request.url ?? "/", "http://127.0.0.1")
    if (url.pathname === "/redirect") {
      response.writeHead(302, { location: "/next" }).end()
      return
    }
    if (url.pathname === "/download") {
      response.writeHead(200, {
          "content-disposition": "attachment; filename=magnitude-browser-fixture.txt",
          "content-type": "text/plain",
      }).end("browser download fixture")
      return
    }
    if (url.pathname === "/favicon.svg") {
      response.writeHead(200, { "content-type": "image/svg+xml" })
        .end('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><circle cx="8" cy="8" r="7" fill="#168de2"/></svg>')
      return
    }
    if (url.pathname === "/slow") {
      setTimeout(() => response.writeHead(200, htmlHeaders).end(html("Slow page", "<p id='slow'>Finished</p>")), 2_000)
      return
    }
    if (url.pathname === "/late-history") {
      response.writeHead(200, htmlHeaders).end(html("Late history", `
        <h1>Late history fixture</h1>
        <script>
          setTimeout(() => history.replaceState(null, "", "/late-history#cobssid=s"), 500)
        </script>
      `))
      return
    }
    if (url.pathname === "/next") {
      response.writeHead(200, htmlHeaders).end(html("Fixture Next", "<h1>Next page</h1><a href='/'>Home</a>"))
      return
    }
    response.writeHead(200, htmlHeaders).end(html("Fixture Home", `
      <h1>Browser fixture</h1>
      <link rel="icon" href="/favicon.svg">
      <a id="next" href="/next">Next</a>
      <a id="popup" href="/next" target="_blank">Popup</a>
      <button id="cookie" onclick="document.cookie='magnitude_browser=shared; SameSite=Lax'">Set cookie</button>
      <button id="clipboard" onclick="navigator.clipboard.readText().then(() => document.body.dataset.permission='allowed', () => document.body.dataset.permission='denied')">Read clipboard</button>
      <button id="notifications" onclick="Notification.requestPermission().then(value => document.body.dataset.permission=value)">Request notifications</button>
      <button id="geolocation" onclick="{
        document.body.dataset.geolocationRequests = 'started';
        for (let index = 0; index < 5; index += 1) {
          navigator.geolocation.getCurrentPosition(() => {}, () => {});
        }
      }">Request geolocation repeatedly</button>
      <input id="form" aria-label="Fixture input">
      <input id="upload" type="file" aria-label="Fixture upload">
      <a id="download" href="/download">Download</a>
    `))
})

const htmlHeaders = { "content-type": "text/html; charset=utf-8" }
function html(title: string, body: string): string {
  return `<!doctype html><html><head><title>${title}</title></head><body>${body}</body></html>`
}

await new Promise<void>((resolve) => fixture.listen(0, "127.0.0.1", resolve))
const fixtureUrl = `http://127.0.0.1:${(fixture.address() as AddressInfo).port}`
const tempRoot = await mkdtemp(join(tmpdir(), "magnitude-browser-smoke-"))
const desktopRoot = dirname(dirname(fileURLToPath(import.meta.url)))
const require = createRequire(import.meta.url)
const electronExecutable = require("electron") as string
let browser: Browser | null = null
let electronProcess: ChildProcess | null = null
const sleep = (milliseconds: number) => new Promise<void>((resolve) => setTimeout(resolve, milliseconds))

async function withTimeout<T>(label: string, promise: Promise<T>, timeout = 10_000): Promise<T> {
  return await Promise.race([
    promise,
    sleep(timeout).then(() => {
      throw new Error(`${label} timed out after ${timeout}ms.`)
    }),
  ])
}

async function waitForPage(
  context: BrowserContext,
  predicate: (page: Page) => boolean,
  timeout = 30_000,
): Promise<Page> {
  const deadline = Date.now() + timeout
  while (Date.now() < deadline) {
    const match = context.pages().find(predicate)
    if (match !== undefined) return match
    await sleep(50)
  }
  throw new Error(`Timed out waiting for an Electron page. Current pages: ${context.pages().map((page) => page.url()).join(", ")}`)
}

async function openBrowser(page: Page): Promise<void> {
  const expandButtons = page.getByRole("button", { name: "Expand sidebar" })
  await expandButtons.last().click()
  await page.getByRole("button", { name: "Browser", exact: true }).click()
  await page.getByRole("textbox", { name: "Address and search" }).waitFor()
}

async function navigate(page: Page, value: string): Promise<void> {
  const address = page.getByRole("textbox", { name: "Address and search" })
  await address.fill(value)
  await address.press("Enter")
}

async function tabCloseButton(page: Page, title: string): Promise<void> {
  const tab = page.getByRole("tablist", { name: "Browser tabs" })
    .getByRole("tab", { name: title })
  await tab.locator("..").locator('[aria-label^="Close "]').evaluate((element: HTMLElement) => element.click())
}

async function waitForFile(path: string, timeout = 30_000): Promise<void> {
  const deadline = Date.now() + timeout
  while (Date.now() < deadline) {
    try {
      await access(path)
      return
    } catch {}
    await sleep(50)
  }
  throw new Error(`Timed out waiting for downloaded file: ${path}`)
}

async function evaluateInElectronMain(port: number, expression: string): Promise<void> {
  const deadline = Date.now() + 30_000
  let inspectorUrl: string | undefined
  while (inspectorUrl === undefined) {
    try {
      const response = await fetch(`http://127.0.0.1:${port}/json/list`)
      if (response.ok) {
        const targets = await response.json() as ReadonlyArray<{ webSocketDebuggerUrl?: string }>
        inspectorUrl = targets[0]?.webSocketDebuggerUrl
      }
    } catch {}
    if (Date.now() >= deadline) throw new Error("Electron main-process inspector did not become available.")
    if (inspectorUrl === undefined) await sleep(50)
  }

  const socket = new WebSocket(inspectorUrl)
  await new Promise<void>((resolve, reject) => {
    socket.addEventListener("open", () => resolve(), { once: true })
    socket.addEventListener("error", () => reject(new Error("Could not connect to Electron's main-process inspector.")), { once: true })
  })
  try {
    const response = await new Promise<{ error?: unknown; result?: { exceptionDetails?: unknown } }>((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error("Electron main-process evaluation timed out.")), 30_000)
      socket.addEventListener("message", (event) => {
        const message = JSON.parse(String(event.data)) as { id?: number; error?: unknown; result?: { exceptionDetails?: unknown } }
        if (message.id !== 1) return
        clearTimeout(timeout)
        resolve(message)
      })
      socket.send(JSON.stringify({
        id: 1,
        method: "Runtime.evaluate",
        params: { expression, awaitPromise: true, returnByValue: true },
      }))
    })
    assert.equal(response.error, undefined, `main-process inspector error: ${JSON.stringify(response.error)}`)
    assert.equal(response.result?.exceptionDetails, undefined, `main-process evaluation failed: ${JSON.stringify(response.result?.exceptionDetails)}`)
  } finally {
    socket.close()
  }
}

try {
  const debuggingPort = 42_000 + Math.floor(Math.random() * 10_000)
  const inspectorPort = 52_000 + Math.floor(Math.random() * 8_000)
  electronProcess = spawn(
      electronExecutable,
    [
      join(desktopRoot, "out", "main", "main.js"),
      `--user-data-dir=${join(tempRoot, "electron")}`,
      `--remote-debugging-port=${debuggingPort}`,
      `--inspect=${inspectorPort}`,
    ], {
      cwd: desktopRoot,
      env: {
        ...process.env,
        ELECTRON_ENABLE_SECURITY_WARNINGS: "true",
      },
      stdio: "inherit",
    },
  )
  const deadline = Date.now() + 30_000
  while (true) {
    try {
      const response = await fetch(`http://127.0.0.1:${debuggingPort}/json/version`)
      if (response.ok) break
    } catch {}
    if (Date.now() >= deadline) throw new Error("Electron did not expose its debugging endpoint.")
    await sleep(100)
  }
  browser = await chromium.connectOverCDP(`http://127.0.0.1:${debuggingPort}`)
  const context = browser.contexts()[0]!
  const qaArtifacts = process.env["MAGNITUDE_BROWSER_QA_ARTIFACTS"]
  if (qaArtifacts !== undefined) await mkdir(qaArtifacts, { recursive: true })
  const page = await waitForPage(context, (candidate) => candidate.url().includes("/renderer/index.html"))
  page.setDefaultTimeout(30_000)
  const consoleErrors: string[] = []
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text())
  })
  page.on("pageerror", (error) => consoleErrors.push(error.message))

  await page.waitForTimeout(3_000)
  console.log(`Initial renderer state: ${(await page.locator("body").innerText()).slice(0, 500)}`)
  await page.getByText("What would you like to do", { exact: false }).waitFor({ timeout: 60_000 })
  await openBrowser(page)
  if (process.env["MAGNITUDE_BROWSER_EXTERNAL_QA"] === "1") {
    await navigate(page, "magnitude browser smoke test")
    const googleGuest = await waitForPage(
      context,
      (candidate) => candidate.url().startsWith("https://www.google.com/search?q=magnitude%20browser%20smoke%20test"),
      30_000,
    )
    await googleGuest.waitForLoadState("domcontentloaded")
    await googleGuest.waitForTimeout(2_000)
    const googleDocument = await googleGuest.evaluate(() => ({
      readyState: document.readyState,
      text: document.body?.innerText.slice(0, 500) ?? "",
    }))
    assert.equal(googleDocument.readyState, "complete")
    assert(googleDocument.text.length > 0, "Google should render a non-empty response")
    await page.getByRole("button", { name: "Reload" }).waitFor()
  }
  const browserResizer = page.getByRole("separator", { name: "Resize browser" })
  const initialBrowserWidth = Number(await browserResizer.getAttribute("aria-valuenow"))
  await browserResizer.press("ArrowRight")
  assert.equal(Number(await browserResizer.getAttribute("aria-valuenow")), initialBrowserWidth - 16)
  await navigate(page, fixtureUrl)
  await page.getByRole("tab", { name: "Fixture Home" }).waitFor()

  const firstGuest = await waitForPage(context, (candidate) => candidate.url() === `${fixtureUrl}/`)
  const initialGuest = await firstGuest.evaluate(() => {
    return {
      url: location.href,
      nodeType: typeof process,
      magnitudeBridgeType: typeof (globalThis as { __magnitudeDesktop?: unknown }).__magnitudeDesktop,
      heading: document.querySelector("h1")?.textContent,
    }
  })
  assert(initialGuest, "the fixture page should have a live guest")
  assert.equal(initialGuest.url, `${fixtureUrl}/`)
  assert.equal(initialGuest.heading, "Browser fixture")
  assert.equal(initialGuest.nodeType, "undefined")
  assert.equal(initialGuest.magnitudeBridgeType, "undefined")

  await page.waitForTimeout(200)
  const browserViewport = await page.locator("[data-browser-viewport]").boundingBox()
  assert(browserViewport, "the browser viewport should be measurable")
  const browserResizeHandle = await page.getByRole("separator", { name: "Resize browser" }).boundingBox()
  assert(browserResizeHandle, "the browser resize handle should be measurable")
  assert(
    Math.abs(browserResizeHandle.x + browserResizeHandle.width - browserViewport.x) <= 1,
    "the browser resize handle should end at the native viewport instead of overlapping it",
  )
  await browserResizer.hover({
    position: { x: browserResizeHandle.width / 2, y: browserResizeHandle.height - 24 },
  })
  await page.waitForFunction(() => {
    const element = document.querySelector<HTMLElement>('[role="separator"][aria-label="Resize browser"]')
    return element !== null && getComputedStyle(element).borderRightColor !== "rgba(0, 0, 0, 0)"
  }, undefined, { timeout: 1_000 }).catch(() => undefined)
  const resizeHoverState = await browserResizer.evaluate((element) => {
    const bounds = element.getBoundingClientRect()
    const point = { x: bounds.left + bounds.width / 2, y: bounds.bottom - 24 }
    const target = document.elementFromPoint(point.x, point.y)
    const style = getComputedStyle(element)
    return {
      hovered: element.matches(":hover"),
      indicatorColor: style.borderRightColor,
      targetRole: target?.getAttribute("role") ?? null,
      targetLabel: target?.getAttribute("aria-label") ?? null,
      targetClassName: typeof target?.className === "string" ? target.className : null,
    }
  })
  assert.notEqual(
    resizeHoverState.indicatorColor,
    "rgba(0, 0, 0, 0)",
    `the resize indicator should appear over loaded page content: ${JSON.stringify(resizeHoverState)}`,
  )
  const expectedGuestBounds = {
    x: Math.round(browserViewport.x),
    y: Math.round(browserViewport.y),
    width: Math.round(browserViewport.width),
    height: Math.round(browserViewport.height),
  }
  await evaluateInElectronMain(inspectorPort, `(() => {
    const { BrowserWindow } = process.getBuiltinModule("module").createRequire(process.cwd() + "/package.json")("electron")
    const active = BrowserWindow.getAllWindows()[0].contentView.children.at(-1)
    const actual = active?.getBounds()
    const expected = ${JSON.stringify(expectedGuestBounds)}
    const settled = actual !== undefined
      && Math.abs(actual.x - expected.x) <= 1
      && Math.abs(actual.y - expected.y) <= 1
      && Math.abs(actual.width - expected.width) <= 1
      && Math.abs(actual.height - expected.height) <= 1
    if (!settled) {
      throw new Error("The native browser view does not fill its viewport: " + JSON.stringify({ actual, expected }))
    }
  })()`)

  await firstGuest.evaluate(() => history.replaceState(null, "", "/#section"))
  await page.waitForFunction(
    (expected) => (document.querySelector('[aria-label="Address and search"]') as HTMLInputElement | null)?.value === expected,
    `${fixtureUrl}/#section`,
  )
  await page.getByRole("button", { name: "Reload" }).waitFor()
  await firstGuest.evaluate(() => history.replaceState(null, "", "/"))

  await firstGuest.locator("#next").click()
  await page.getByRole("tab", { name: "Fixture Next" }).waitFor()
  await page.getByRole("button", { name: "Back" }).click()
  await page.getByRole("tab", { name: "Fixture Home" }).waitFor()
  await page.getByRole("button", { name: "Forward" }).click()
  await page.getByRole("tab", { name: "Fixture Next" }).waitFor()

  await page.getByRole("button", { name: "New tab" }).click()
  await navigate(page, fixtureUrl)
  await page.getByRole("tab", { name: "Fixture Home" }).waitFor()
  await tabCloseButton(page, "Fixture Home")
  await page.getByRole("tab", { name: "Fixture Next" }).waitFor()
  await evaluateInElectronMain(inspectorPort, `(() => {
    const { BrowserWindow } = process.getBuiltinModule("module").createRequire(process.cwd() + "/package.json")("electron")
    const children = BrowserWindow.getAllWindows()[0].contentView.children
    const active = children.at(-1)
    if (active?.webContents?.getURL() !== ${JSON.stringify(`${fixtureUrl}/next`)} || !active.getVisible()) {
      throw new Error("The retained browser tab was not restored as the visible topmost view.")
    }
  })()`)

  await page.getByRole("button", { name: "New tab" }).click()
  const browserTabs = page.getByRole("tablist", { name: "Browser tabs" }).getByRole("tab")
  assert.equal(await browserTabs.count(), 2)
  assert.equal(
    await page.getByRole("textbox", { name: "Address and search" }).evaluate((element) => element === document.activeElement),
    true,
  )
  await navigate(page, fixtureUrl)
  const cookieGuest = await waitForPage(context, (candidate) => candidate.url() === `${fixtureUrl}/`)
  await cookieGuest.locator("#cookie").click()
  await page.getByRole("button", { name: "New tab" }).click()
  await navigate(page, fixtureUrl)
  const sharedCookieGuest = context.pages().filter((candidate) => candidate.url() === `${fixtureUrl}/`).at(-1)!
  const sharedCookie = await sharedCookieGuest.evaluate(() => document.cookie)
  assert.match(sharedCookie, /magnitude_browser=shared/)

  await sharedCookieGuest.locator("#form").fill("kept in the guest")
  assert.equal(await sharedCookieGuest.locator("#form").inputValue(), "kept in the guest")
  const uploadPath = join(tempRoot, "upload-fixture.md")
  await writeFile(uploadPath, "# Uploaded through the embedded browser\n")
  await sharedCookieGuest.locator("#upload").setInputFiles(uploadPath)
  const uploaded = await sharedCookieGuest.locator("#upload").evaluate(async (element: HTMLInputElement) => {
    const file = element.files?.[0]
    return file === undefined ? null : { name: file.name, contents: await file.text() }
  })
  assert.deepEqual(uploaded, {
    name: "upload-fixture.md",
    contents: "# Uploaded through the embedded browser\n",
  })

  await browserTabs.last().focus()
  await browserTabs.last().press("ArrowLeft")
  await expectSelected(browserTabs.nth(1))
  await browserTabs.last().click()
  await sharedCookieGuest.locator("#notifications").click()
  await page.getByText("Allow site permission?").waitFor()
  assert.match(await page.getByRole("alertdialog").innerText(), /127\.0\.0\.1/)
  if (qaArtifacts !== undefined) {
    await page.waitForTimeout(150)
    await page.screenshot({ path: join(qaArtifacts, "permission-dialog.png") })
  }
  await page.getByRole("button", { name: "Deny" }).evaluate((element: HTMLElement) => element.click())
  await page.getByText("Allow site permission?").waitFor({ state: "hidden" })
  await sharedCookieGuest.waitForFunction(() => document.body.dataset.permission === "denied")

  await navigate(page, `${fixtureUrl}/redirect`)
  await page.getByRole("tab", { name: "Fixture Next" }).last().waitFor()
  await navigate(page, `${fixtureUrl}/late-history`)
  await page.getByRole("tab", { name: "Late history" }).waitFor()
  await navigate(page, `${fixtureUrl}/slow`)
  await page.waitForTimeout(750)
  assert.equal(
    await page.getByRole("textbox", { name: "Address and search" }).inputValue(),
    `${fixtureUrl}/slow`,
    "a stale in-page navigation must not overwrite a newer document navigation",
  )
  await page.getByRole("button", { name: "Stop" }).waitFor()
  await page.getByRole("tab", { name: "Slow page" }).waitFor()
  await page.getByRole("button", { name: "Reload" }).waitFor()
  await navigate(page, `${fixtureUrl}/slow`)
  await page.getByRole("button", { name: "Stop" }).click()
  await page.getByRole("button", { name: "Reload" }).waitFor()
  await navigate(page, fixtureUrl)
  const activeFixtureGuest = context.pages().filter((candidate) => candidate.url() === `${fixtureUrl}/`).at(-1)!
  await activeFixtureGuest.locator("#notifications").click()
  await page.getByText("Allow site permission?").waitFor()
  await page.getByRole("button", { name: "Allow once" }).evaluate((element: HTMLElement) => element.click())
  await page.getByText("Allow site permission?").waitFor({ state: "hidden" })
  await activeFixtureGuest.waitForFunction(() => document.body.dataset.permission === "granted")

  await activeFixtureGuest.locator("#geolocation").click()
  await page.getByText("Allow site permission?").waitFor()
  assert.match(await page.getByRole("alertdialog").innerText(), /geolocation/)
  await page.getByRole("button", { name: "Allow once" }).evaluate((element: HTMLElement) => element.click())
  await page.getByText("Allow site permission?").waitFor({ state: "hidden" })
  await page.waitForTimeout(500)
  assert.equal(
    await page.getByText("Allow site permission?").isVisible(),
    false,
    "concurrent identical permission requests must share one decision prompt",
  )

  const downloadRoot = join(tempRoot, "downloads")
  await mkdir(downloadRoot, { recursive: true })
  const downloadedFile = join(downloadRoot, "magnitude-browser-fixture.txt")
  await evaluateInElectronMain(inspectorPort, `process.getBuiltinModule("module").createRequire(process.cwd() + "/package.json")("electron").session.fromPartition("persist:magnitude-browser").once("will-download", (_event, item) => item.setSavePath(${JSON.stringify(downloadedFile)}))`)
  await activeFixtureGuest.locator("#download").click()
  await waitForFile(downloadedFile)
  assert.equal(await readFile(downloadedFile, "utf8"), "browser download fixture")
  await page.getByText("Downloaded").waitFor()

  await activeFixtureGuest.locator("#popup").click()
  await page.getByRole("tab", { name: "Fixture Next" }).last().waitFor()
  assert.equal(await browserTabs.count(), 4)
  const magnitudeWindowCount = context.pages().filter((candidate) => candidate.url().includes("/renderer/index.html")).length
  assert.equal(magnitudeWindowCount, 1, "target=_blank must not create another Magnitude window")

  await navigate(page, "file:///etc/passwd")
  await page.getByText("Couldn’t open this page").waitFor()
  const privilegedGuest = context.pages().some((candidate) => candidate.url().startsWith("file:///etc/passwd"))
  assert.equal(privilegedGuest, false)

  await navigate(page, "http://example.com")
  await page.getByText("Insecure connection").waitFor()
  if (qaArtifacts !== undefined) await page.screenshot({ path: join(qaArtifacts, "insecure-interstitial.png") })
  await page.getByRole("button", { name: "Cancel" }).evaluate((element: HTMLElement) => element.click())
  await page.getByText("Insecure connection").waitFor({ state: "hidden" })
  await page.getByRole("tab", { name: "Fixture Next" }).last().waitFor()

  const tabCloseButtons = page.getByRole("tablist", { name: "Browser tabs" }).locator('[aria-label^="Close "]')
  while (await browserTabs.count() > 1) await tabCloseButtons.last().click()
  await tabCloseButtons.last().click()
  assert.equal(await browserTabs.count(), 1)
  await page.getByRole("tab", { name: "New tab" }).waitFor()
  assert.equal(
    await page.getByRole("textbox", { name: "Address and search" }).evaluate((element) => element === document.activeElement),
    true,
  )
  if (qaArtifacts !== undefined) await page.screenshot({ path: join(qaArtifacts, "blank-browser.png") })

  await navigate(page, fixtureUrl)
  await page.getByRole("tab", { name: "Fixture Home" }).waitFor()
  const crashGuest = context.pages().filter((candidate) => candidate.url() === `${fixtureUrl}/`).at(-1)!
  const crashSession = await context.newCDPSession(crashGuest)
  // Chromium commonly terminates the target before acknowledging Page.crash.
  // Bound the protocol call and assert the observable Electron recovery state instead.
  await withTimeout("Crashing the embedded guest", crashSession.send("Page.crash"), 5_000).catch(() => undefined)
  await page.getByText("This page crashed", { exact: true }).waitFor()
  await page.getByRole("button", { name: "Retry" }).click()
  await page.getByRole("tab", { name: "Fixture Home" }).waitFor()

  await page.emulateMedia({ reducedMotion: "reduce" })
  await page.getByRole("button", { name: "Close sidebar" }).click()
  await page.getByRole("button", { name: "Expand sidebar" }).last().waitFor()
  await openBrowser(page)
  await page.getByRole("tab", { name: "Fixture Home" }).waitFor()

  await page.getByRole("button", { name: "Settings", exact: true }).first().click()
  const themeButton = page.getByRole("button", { name: /^Theme:/ })
  await themeButton.waitFor()
  for (let attempt = 0; attempt < 3; attempt += 1) {
    if (await page.locator("html").getAttribute("data-theme") === "dark") break
    await themeButton.click()
  }
  assert.equal(await page.locator("html").getAttribute("data-theme"), "dark")
  await page.getByRole("button", { name: "Settings", exact: true }).first().click()
  if (qaArtifacts !== undefined) await page.screenshot({ path: join(qaArtifacts, "dark-browser.png") })

  const meaningfulErrors = consoleErrors.filter((message) =>
    !message.includes("Download the React DevTools")
    && !message.includes("favicon"),
  )
  assert.deepEqual(meaningfulErrors, [], `renderer errors:\n${meaningfulErrors.join("\n")}`)
  console.log("Embedded browser Electron smoke test passed")
} finally {
  if (electronProcess !== null) {
    const exited = electronProcess.exitCode === null
      ? new Promise<void>((resolve) => electronProcess?.once("exit", () => resolve()))
      : Promise.resolve()
    electronProcess.kill()
    const graceful = await Promise.race([exited.then(() => true), sleep(5_000).then(() => false)])
    if (!graceful && electronProcess.exitCode === null) {
      electronProcess.kill("SIGKILL")
      await Promise.race([exited, sleep(1_000)])
    }
  }
  await Promise.race([browser?.close().catch(() => undefined) ?? Promise.resolve(), sleep(1_000)])
  fixture.closeAllConnections()
  await new Promise<void>((resolve) => fixture.close(() => resolve()))
  await rm(tempRoot, { recursive: true, force: true })
}

async function expectSelected(tab: ReturnType<Page["locator"]>): Promise<void> {
  const deadline = Date.now() + 30_000
  while (Date.now() < deadline) {
    if (await tab.getAttribute("aria-selected") === "true") return
    await sleep(50)
  }
  throw new Error("Timed out waiting for browser tab selection.")
}
