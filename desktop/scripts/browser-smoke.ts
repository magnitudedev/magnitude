import { strict as assert } from "node:assert"
import { spawn, type ChildProcess } from "node:child_process"
import { access, copyFile, cp, mkdir, mkdtemp, readFile, readdir, rm, symlink, writeFile } from "node:fs/promises"
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
const attachmentRoot = join(tempRoot, "composer-attachments")
await mkdir(attachmentRoot, { recursive: true })
const attachmentPaths = {
  markdown: join(attachmentRoot, "notes.md"),
  typescript: join(attachmentRoot, "answer.ts"),
  json: join(attachmentRoot, "config.json"),
  svg: join(attachmentRoot, "icon.svg"),
  image: join(attachmentRoot, "pixel.png"),
  binary: join(attachmentRoot, "binary.bin"),
  oversized: join(attachmentRoot, "oversized.txt"),
}
await Promise.all([
  writeFile(attachmentPaths.markdown, "# Notes\n\nComposer attachment QA.\n"),
  writeFile(attachmentPaths.typescript, "export const answer = 42\n"),
  writeFile(attachmentPaths.json, '{"attachment":true}\n'),
  writeFile(attachmentPaths.svg, '<svg xmlns="http://www.w3.org/2000/svg"><circle cx="4" cy="4" r="3" /></svg>\n'),
  writeFile(
    attachmentPaths.image,
    Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=", "base64"),
  ),
  writeFile(attachmentPaths.binary, Buffer.from([0x61, 0x00, 0x62])),
  writeFile(attachmentPaths.oversized, Buffer.alloc(500 * 1024 + 1, 0x61)),
])

const composerSendQa = process.env["MAGNITUDE_COMPOSER_SEND_QA"] === "1"
const workspaceQa = process.env["MAGNITUDE_WORKSPACE_QA"] === "1"
const isolatedProjectQa = composerSendQa || workspaceQa
const qaDataDir = join(tempRoot, "magnitude-data")
if (isolatedProjectQa) {
  await mkdir(join(attachmentRoot, "nested"), { recursive: true })
  await Promise.all([
    writeFile(join(attachmentRoot, "nested", "nested.ts"), "export const nested = true\n"),
    writeFile(join(attachmentRoot, "delete-me.txt"), "delete this file\n"),
  ])
  const sourceDataDir = join(process.env["HOME"] ?? "", ".magnitude")
  await mkdir(join(qaDataDir, "state"), { recursive: true })
  await Promise.all([
    copyFile(join(sourceDataDir, "state", "models.json"), join(qaDataDir, "state", "models.json")),
    copyFile(join(sourceDataDir, "state", "onboarding.json"), join(qaDataDir, "state", "onboarding.json")),
    copyFile(join(sourceDataDir, "config.json"), join(qaDataDir, "config.json")),
    cp(join(sourceDataDir, "model-catalog"), join(qaDataDir, "model-catalog"), { recursive: true }),
    cp(join(sourceDataDir, "local-inference"), join(qaDataDir, "local-inference"), { recursive: true }),
    cp(join(sourceDataDir, "llamacpp"), join(qaDataDir, "llamacpp"), { recursive: true }),
  ])
  const sourceModelHub = join(sourceDataDir, "models", "hub")
  const qaModelHub = join(qaDataDir, "models", "hub")
  await mkdir(qaModelHub, { recursive: true })
  await copyFile(
    join(sourceDataDir, "models", "catalog-affiliations.json"),
    join(qaDataDir, "models", "catalog-affiliations.json"),
  )
  await Promise.all((await readdir(sourceModelHub)).map((entry) =>
    symlink(join(sourceModelHub, entry), join(qaModelHub, entry), "dir")
  ))
  const now = Date.now()
  await writeFile(join(qaDataDir, "state", "projects.json"), JSON.stringify({
    projects: [{
      projectId: "attachment-qa-project",
      name: "composer-attachments",
      cwd: attachmentRoot,
      registrationState: "active",
      createdAt: now,
      updatedAt: now,
    }],
  }))
}
const desktopRoot = dirname(dirname(fileURLToPath(import.meta.url)))
const require = createRequire(import.meta.url)
const electronExecutable = require("electron") as string
let browser: Browser | null = null
let electronProcess: ChildProcess | null = null
let electronInspectorPort: number | null = null
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
  const address = page.getByRole("textbox", { name: "Address and search" })
  if (await address.isVisible().catch(() => false)) return
  const expandButtons = page.getByRole("button", { name: "Expand sidebar" })
  if (await expandButtons.last().isVisible().catch(() => false)) {
    await expandButtons.last().click()
    await page.waitForTimeout(200)
    assert.equal(
      await page.getByRole("tooltip").filter({ hasText: "Collapse sidebar" }).count(),
      0,
      "the moving panel-header action must not flash a tooltip during expansion",
    )
  }
  if (!await address.isVisible().catch(() => false)) await newBrowserTab(page)
  await page.getByRole("textbox", { name: "Address and search" }).waitFor()
}

async function newBrowserTab(page: Page): Promise<void> {
  await page.getByRole("button", { name: "New workspace tab" }).click()
  await page.getByRole("menuitem", { name: "Browser", exact: true }).click()
  await page.getByRole("textbox", { name: "Address and search" }).waitFor()
}

async function navigate(page: Page, value: string): Promise<void> {
  const address = page.getByRole("textbox", { name: "Address and search" })
  await address.fill(value)
  await address.press("Enter")
}

async function tabCloseButton(page: Page, title: string): Promise<void> {
  const tab = page.getByRole("tablist", { name: "Workspace tabs" })
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

async function signalFixtureProcesses(signal: "TERM" | "KILL"): Promise<void> {
  const process = spawn("pkill", [`-${signal}`, "-f", tempRoot], { stdio: "ignore" })
  await new Promise<void>((resolve) => process.once("exit", () => resolve()))
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
  electronInspectorPort = inspectorPort
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
        ...(isolatedProjectQa ? { MAGNITUDE_DEV_DATA_DIR: qaDataDir } : {}),
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
  await page.getByText("What would you like to do", { exact: false }).waitFor({ timeout: 180_000 })

  // General settings are part of the real Electron renderer and persist
  // through the desktop storage bridge, not browser-local presentation state.
  await page.getByLabel("Settings", { exact: true }).click()
  await page.getByRole("heading", { name: "General" }).waitFor()
  const showThinking = page.getByRole("switch", { name: "Show thinking" })
  assert.equal(await showThinking.isChecked(), false)
  await showThinking.click()
  assert.equal(await showThinking.isChecked(), true)
  await page.waitForTimeout(150)
  let restoredShowThinking = showThinking
  if (!composerSendQa) {
    await page.reload()
    await page.getByText("What would you like to do", { exact: false }).waitFor({ timeout: 60_000 })
    await page.getByLabel("Settings", { exact: true }).click()
    await page.getByRole("heading", { name: "General" }).waitFor({ timeout: 60_000 })
    restoredShowThinking = page.getByRole("switch", { name: "Show thinking" })
    assert.equal(
      await restoredShowThinking.isChecked(),
      true,
      "the desktop Show thinking preference must survive a renderer reload",
    )
  }
  if (qaArtifacts !== undefined) {
    await page.emulateMedia({ colorScheme: "light" })
    await page.waitForTimeout(100)
    await page.screenshot({ path: join(qaArtifacts, "general-settings-light.png") })
    await page.emulateMedia({ colorScheme: "dark" })
    await page.waitForTimeout(100)
    await page.screenshot({ path: join(qaArtifacts, "general-settings-dark.png") })
    await page.emulateMedia({ colorScheme: "light" })
  }
  if (!composerSendQa) await restoredShowThinking.click()
  await page.getByLabel("Settings", { exact: true }).click()
  await page.getByRole("button", { name: "Attach files" }).waitFor()

  // Composer attachment ingestion runs in the real Electron renderer. Exercise
  // the native chooser bridge plus browser drag/drop and paste entry points
  // before opening the native embedded-browser panel.
  const attachButton = page.getByRole("button", { name: "Attach files" })
  const chooserPromise = page.waitForEvent("filechooser")
  await attachButton.click()
  const chooser = await chooserPromise
  assert.equal(chooser.isMultiple(), true)
  await chooser.setFiles([
    attachmentPaths.markdown,
    attachmentPaths.typescript,
    attachmentPaths.json,
    attachmentPaths.svg,
    attachmentPaths.image,
  ])

  const attachmentRow = page.getByLabel("Attached files")
  await attachmentRow.waitFor()
  for (const filename of ["notes.md", "answer.ts", "config.json", "icon.svg", "pixel.png"]) {
    await attachmentRow.getByLabel(`Reading ${filename}`).waitFor({ state: "detached" })
    await attachmentRow.getByText(filename).waitFor()
  }
  assert.equal(await attachmentRow.locator(":scope > div").count(), 5)
  const rowBox = await attachmentRow.boundingBox()
  assert(
    rowBox !== null && Math.abs(rowBox.height - 72) <= 1,
    `the attachment row must keep its fixed 72px height: ${JSON.stringify(rowBox)}`,
  )
  if (qaArtifacts !== undefined) {
    await page.emulateMedia({ colorScheme: "light" })
    await page.waitForTimeout(100)
    await page.screenshot({ path: join(qaArtifacts, "composer-attachments-light.png") })
    await page.emulateMedia({ colorScheme: "dark" })
    await page.waitForTimeout(100)
    await page.screenshot({ path: join(qaArtifacts, "composer-attachments-dark.png") })
    await page.emulateMedia({ colorScheme: "light" })
  }

  await page.getByRole("button", { name: "Remove answer.ts" }).click()
  await page.getByRole("button", { name: "Remove answer.ts" }).waitFor({ state: "detached" })
  assert.equal(await attachmentRow.locator(":scope > div").count(), 4)

  const composerSurface = page.locator(".composer > div").first()
  await composerSurface.evaluate((element) => {
    const transfer = new DataTransfer()
    transfer.items.add(new File(["print('dragged')\n"], "dragged.py", { type: "text/x-python" }))
    element.dispatchEvent(new DragEvent("dragenter", { bubbles: true, cancelable: true, dataTransfer: transfer }))
    element.dispatchEvent(new DragEvent("dragover", { bubbles: true, cancelable: true, dataTransfer: transfer }))
    element.dispatchEvent(new DragEvent("drop", { bubbles: true, cancelable: true, dataTransfer: transfer }))
  })
  await attachmentRow.getByLabel("Reading dragged.py").waitFor({ state: "detached" })
  await attachmentRow.getByText("dragged.py").waitFor()

  await page.getByRole("textbox", { name: "Message" }).evaluate((element) => {
    const transfer = new DataTransfer()
    transfer.items.add(new File(["pasted=true\n"], "pasted.env", { type: "text/plain" }))
    element.dispatchEvent(new ClipboardEvent("paste", { bubbles: true, cancelable: true, clipboardData: transfer }))
  })
  await attachmentRow.getByLabel("Reading pasted.env").waitFor({ state: "detached" })
  await attachmentRow.getByText("pasted.env").waitFor()

  await page.locator('input[type="file"]').setInputFiles(attachmentPaths.binary)
  await page.getByRole("alert").getByText("binary.bin: Binary files are not supported.").waitFor()
  await page.locator('input[type="file"]').setInputFiles(attachmentPaths.oversized)
  await page.getByRole("alert").getByText("oversized.txt: Text and code files must be 500 KiB or smaller.").waitFor()
  const attachmentAlertClose = page.getByRole("alert").getByRole("button")
  while (await attachmentAlertClose.count() > 0) await attachmentAlertClose.first().click()

  if (composerSendQa) {
    for (const filename of ["config.json", "icon.svg", "pixel.png", "dragged.py", "pasted.env"]) {
      await page.getByRole("button", { name: `Remove ${filename}` }).click()
    }
    assert.equal(await attachmentRow.locator(":scope > div").count(), 1)
    await page.getByRole("textbox", { name: "Message" }).fill(
      "Read the attached notes and reply with one short sentence summarizing them.",
    )
    const workSummary = page.locator(".chat-timeline").getByText(/worked for/i)
    const initialWorkSummaryCount = await workSummary.count()
    const inlineActivitySeen = page.waitForFunction(() => {
      const text = document.querySelector(".chat-timeline")?.textContent ?? ""
      return /Loading .+|Preparing context|Working|Thinking/.test(text)
    }, undefined, { timeout: 60_000 })
    await page.getByRole("button", { name: "Send message" }).click()
    await inlineActivitySeen.catch(async (error) => {
      console.error(`Composer-send state after timeout: ${(await page.locator("body").innerText()).slice(0, 2_000)}`)
      if (qaArtifacts !== undefined) {
        await page.screenshot({ path: join(qaArtifacts, "inline-agent-activity-timeout.png") })
      }
      throw error
    })
    if (qaArtifacts !== undefined) {
      await page.screenshot({ path: join(qaArtifacts, "inline-agent-activity.png") })
    }
    await attachmentRow.waitFor({ state: "detached" })
    await page.getByText("notes.md", { exact: true }).waitFor()
    const progressSamples: number[] = []
    const activityDeadline = Date.now() + 600_000
    while (Date.now() < activityDeadline
      && await workSummary.count() <= initialWorkSummaryCount) {
      const activity = page.locator('[data-activity-kind="model-loading"]')
      if (await activity.count() > 0) {
        const progressbar = activity.getByRole("progressbar")
        if (await progressbar.count() > 0) {
          const raw = await progressbar.getAttribute("aria-valuenow")
          if (raw !== null) progressSamples.push(Number(raw))
        }
      }
      await page.waitForTimeout(50)
    }
    await workSummary.nth(initialWorkSummaryCount).waitFor({ timeout: 600_000 })
    assert(
      progressSamples.some((value) => value > 0),
      `the authoritative model-load activity never advanced beyond zero: ${JSON.stringify(progressSamples)}`,
    )
    await page.locator('[data-activity-kind="model-loading"]').waitFor({ state: "detached" })
    if (qaArtifacts !== undefined) {
      await page.screenshot({ path: join(qaArtifacts, "completed-agent-activity.png") })
    }

    const sessionsRoot = join(qaDataDir, "sessions")
    await withTimeout("attachment message persistence", (async () => {
      while (true) {
        const entries = await readdir(sessionsRoot, { withFileTypes: true }).catch(() => [])
        for (const entry of entries) {
          if (!entry.isDirectory() || entry.name === "index") continue
          const captured = join(sessionsRoot, entry.name, "scratchpad", "attachments", "notes.md")
          try {
            assert.equal(await readFile(captured, "utf8"), "# Notes\n\nComposer attachment QA.\n")
            return
          } catch {}
        }
        await sleep(50)
      }
    })(), 30_000)
  }

  const removeButtons = page.getByRole("button", { name: /^Remove / })
  while (await removeButtons.count() > 0) await removeButtons.first().click()
  await attachmentRow.waitFor({ state: "detached" })

  if (workspaceQa) {
    await evaluateInElectronMain(inspectorPort, `(() => {
      const { BrowserWindow } = process.getBuiltinModule("module").createRequire(process.cwd() + "/package.json")("electron")
      BrowserWindow.getAllWindows()[0].setSize(1_600, 900)
    })()`)
    await page.waitForFunction(() => window.innerWidth >= 1_590)
    await page.getByRole("button", { name: "Expand sidebar" }).last().click()
    const workspace = page.getByRole("complementary", { name: "Workspace" })
    await workspace.waitFor()
    await page.waitForTimeout(250)
    const fileTabs = page.getByRole("tablist", { name: "Workspace tabs" })
      .locator('[role="tab"][data-workspace-tab-kind="file"]')
    assert.equal(await fileTabs.count(), 1, "opening the workspace for a project should create one empty file tab")
    await page.getByText("Select a file", { exact: true }).last().waitFor()
    await page.getByRole("tree", { name: "Project file tree" }).waitFor()

    const workspaceBox = await workspace.boundingBox()
    const treeDock = page.getByRole("region", { name: "Project files" })
    const initialTreeBox = await treeDock.boundingBox()
    assert(workspaceBox !== null && Math.abs(workspaceBox.width - 600) <= 2, `workspace should begin at 600px: ${JSON.stringify(workspaceBox)}`)
    assert(initialTreeBox !== null && Math.abs(initialTreeBox.width - 200) <= 2, `project tree should begin at 200px: ${JSON.stringify(initialTreeBox)}`)

    await page.getByText("nested", { exact: true }).click()
    await page.getByText("nested.ts", { exact: true }).waitFor()
    const externalFile = join(attachmentRoot, "external-change.txt")
    await writeFile(externalFile, "created outside Magnitude\n")
    await page.getByText("external-change.txt", { exact: true }).waitFor()
    await rm(externalFile)
    await page.getByText("external-change.txt", { exact: true }).waitFor({ state: "detached" })

    await page.getByText("answer.ts", { exact: true }).click()
    await page.getByRole("tab", { name: "answer.ts" }).waitFor()
    assert.equal(await fileTabs.count(), 1, "the first tree selection should fill the active file tab")
    await page.getByText("nested.ts", { exact: true }).click()
    await page.getByRole("tab", { name: "nested.ts" }).waitFor()
    assert.equal(await fileTabs.count(), 1, "later tree selections should replace the active file tab")
    await page.getByText("answer.ts", { exact: true }).click()
    await page.getByRole("tab", { name: "answer.ts" }).waitFor()
    assert.equal(await fileTabs.count(), 1, "tree selection should not create implicit file tabs")
    const monacoEditor = page.locator(".monaco-editor").last()
    const saveButton = page.getByRole("button", { name: "Save", exact: true })
    await monacoEditor.waitFor()
    await monacoEditor.click()
    await page.keyboard.press(process.platform === "darwin" ? "Meta+A" : "Control+A")
    await page.keyboard.insertText("export const broken =\n")
    await withTimeout("showing syntax diagnostics", (async () => {
      while (await monacoEditor.locator(".squiggly-error").count() === 0) await sleep(25)
    })())
    await page.keyboard.press(process.platform === "darwin" ? "Meta+A" : "Control+A")
    await page.keyboard.insertText('import { Effect } from "effect"\nexport const result = Effect.succeed(42)\n')
    await withTimeout("clearing syntax diagnostics", (async () => {
      while (await monacoEditor.locator(".squiggly-error").count() > 0) await sleep(25)
    })())
    await sleep(750)
    assert(await monacoEditor.locator(".squiggly-error").count() === 0, "project-blind Monaco worker should not report unresolved external imports")
    await page.keyboard.press(process.platform === "darwin" ? "Meta+A" : "Control+A")
    await page.keyboard.insertText("export const updated = 84\n")
    await withTimeout("enabling save after editing", (async () => {
      while (!(await saveButton.isEnabled())) await sleep(25)
    })())

    await fileTabs.last().locator("..").locator('[aria-label="Close answer.ts"]').click()
    await page.getByRole("alertdialog").getByText("Discard unsaved changes?").waitFor()
    await page.getByRole("button", { name: "Keep editing" }).click()
    await saveButton.click()
    await withTimeout("saving an edited project file", (async () => {
      while (await readFile(attachmentPaths.typescript, "utf8") !== "export const updated = 84\n") await sleep(25)
    })())
    await withTimeout("disabling save after persistence", (async () => {
      while (await saveButton.isEnabled()) await sleep(25)
    })())

    await monacoEditor.click()
    await page.keyboard.press(process.platform === "darwin" ? "Meta+A" : "Control+A")
    await page.keyboard.insertText("export const draftAcrossTabs = 126\n")
    await withTimeout("enabling save for a second edit", (async () => {
      while (!(await saveButton.isEnabled())) await sleep(25)
    })())

    await page.getByText("notes.md", { exact: true }).click()
    await page.getByRole("alertdialog").getByText("Discard unsaved changes?").waitFor()
    await page.getByRole("button", { name: "Keep editing" }).click()
    await page.getByRole("tab", { name: "answer.ts" }).waitFor()
    await page.locator(".view-lines").last().getByText(/draftAcrossTabs/).waitFor()

    await page.getByRole("button", { name: "New workspace tab" }).click()
    await page.getByRole("menuitem", { name: "File", exact: true }).click()
    await page.getByText("notes.md", { exact: true }).click()
    await page.getByRole("tab", { name: "notes.md" }).waitFor()
    await page.getByRole("button", { name: "Preview", exact: true }).waitFor()
    await page.getByRole("heading", { name: "Notes", exact: true }).waitFor()
    await page.getByRole("button", { name: "Source", exact: true }).click()
    await page.locator(".monaco-editor").last().waitFor()
    await page.getByRole("tab", { name: "answer.ts" }).click()
    await withTimeout("preserving the dirty draft across tabs", (async () => {
      while (!(await saveButton.isEnabled())) await sleep(25)
    })())
    await page.locator(".view-lines").last().getByText(/draftAcrossTabs/).waitFor()
    const tabSurfaceStyles = await page.evaluate(() => {
      const tabs = [...document.querySelectorAll<HTMLElement>('[role="tab"]')]
      const answer = tabs.find((tab) => tab.textContent?.includes("answer.ts"))?.parentElement
      const notes = tabs.find((tab) => tab.textContent?.includes("notes.md"))?.parentElement
      const header = answer?.closest("header")
      if (answer === undefined || answer === null || notes === undefined || notes === null || header === null || header === undefined) return null
      const active = getComputedStyle(answer)
      const inactive = getComputedStyle(notes)
      const container = getComputedStyle(header)
      return {
        activeBackground: active.backgroundColor,
        inactiveBackground: inactive.backgroundColor,
        containerBackground: container.backgroundColor,
        activeRadius: Number.parseFloat(active.borderRadius),
      }
    })
    assert(tabSurfaceStyles !== null, "workspace tab surfaces should be rendered")
    assert(tabSurfaceStyles.activeBackground !== tabSurfaceStyles.inactiveBackground, `active and inactive tabs should have distinct surfaces: ${JSON.stringify(tabSurfaceStyles)}`)
    assert(tabSurfaceStyles.inactiveBackground !== tabSurfaceStyles.containerBackground, `tabs should be distinct from the header: ${JSON.stringify(tabSurfaceStyles)}`)
    assert(tabSurfaceStyles.activeRadius >= 6, `workspace tabs should be rounded: ${JSON.stringify(tabSurfaceStyles)}`)
    await page.getByRole("treeitem").filter({ hasText: "answer.ts" }).waitFor({ state: "visible" })
    await withTimeout("synchronizing the project tree selection with the active file tab", (async () => {
      const answerTreeItem = page.getByRole("treeitem").filter({ hasText: "answer.ts" })
      while (await answerTreeItem.getAttribute("aria-selected") !== "true") await sleep(25)
    })())
    if (qaArtifacts !== undefined) {
      await page.emulateMedia({ colorScheme: "light" })
      await page.screenshot({ path: join(qaArtifacts, "workspace-editor-light.png") })
      await page.emulateMedia({ colorScheme: "dark" })
      await page.screenshot({ path: join(qaArtifacts, "workspace-editor-dark.png") })
      await page.emulateMedia({ colorScheme: "light" })
    }
    await saveButton.click()
    await withTimeout("saving a draft after tab switching", (async () => {
      while (await readFile(attachmentPaths.typescript, "utf8") !== "export const draftAcrossTabs = 126\n") await sleep(25)
    })())

    await monacoEditor.click()
    await page.keyboard.press(process.platform === "darwin" ? "Meta+A" : "Control+A")
    await page.keyboard.insertText("export const discarded = 999\n")
    await page.getByText("pixel.png", { exact: true }).click()
    await page.getByRole("alertdialog").getByText("Discard unsaved changes?").waitFor()
    await page.getByRole("button", { name: "Discard changes" }).click()
    await page.getByRole("tab", { name: "pixel.png" }).waitFor()
    await page.getByRole("img", { name: "pixel.png" }).waitFor()
    await page.getByText("answer.ts", { exact: true }).click()
    await page.getByRole("tab", { name: "answer.ts" }).waitFor()
    await page.locator(".view-lines").last().getByText(/draftAcrossTabs/).waitFor()
    assert.equal(await page.locator(".view-lines").last().getByText(/discarded/).count(), 0, "discarded edits must not return when the file is reopened")
    await page.getByText("pixel.png", { exact: true }).click()
    await page.getByRole("img", { name: "pixel.png" }).waitFor()
    await page.getByText("binary.bin", { exact: true }).click()
    await page.getByRole("tab", { name: "binary.bin" }).waitFor()
    await page.getByText(/This file cannot be displayed/).waitFor()

    await page.getByText("delete-me.txt", { exact: true }).click()
    await page.getByRole("tab", { name: "delete-me.txt" }).waitFor()
    await page.getByRole("button", { name: "File actions" }).click()
    await page.getByRole("menuitem", { name: "Remove file" }).click()
    await page.getByRole("alertdialog").getByText("Remove this file?").waitFor()
    await page.getByRole("button", { name: "Remove", exact: true }).click()
    await withTimeout("deleting a project file", (async () => {
      while (true) {
        try {
          await access(join(attachmentRoot, "delete-me.txt"))
        } catch {
          return
        }
        await sleep(25)
      }
    })())
    await page.getByRole("tab", { name: "delete-me.txt" }).waitFor({ state: "detached" })

    const tabListBox = await page.getByRole("tablist", { name: "Workspace tabs" }).boundingBox()
    const newTabButton = page.getByRole("button", { name: "New workspace tab" })
    const newTabBox = await newTabButton.boundingBox()
    assert(tabListBox !== null && newTabBox !== null && newTabBox.x + newTabBox.width <= tabListBox.x + tabListBox.width + 1, "the new-tab action must remain visible when tabs overflow")
    await newTabButton.click()
    await page.getByRole("menuitem", { name: "File", exact: true }).click()
    await page.getByText("Select a file", { exact: true }).last().waitFor()
    const panelBeforeTreeToggle = await workspace.boundingBox()
    await page.getByRole("button", { name: "Hide project files" }).click()
    await treeDock.waitFor({ state: "detached" })
    const panelAfterTreeToggle = await workspace.boundingBox()
    assert.equal(Math.round(panelAfterTreeToggle?.width ?? 0), Math.round(panelBeforeTreeToggle?.width ?? 0), "the tree must resize content inside the workspace instead of growing the panel")
    await page.getByRole("button", { name: "Show project files" }).click()
    await treeDock.waitFor()
    const treeResizer = page.getByRole("separator", { name: "Resize project files" })
    const treeWidth = Number(await treeResizer.getAttribute("aria-valuenow"))
    await treeResizer.press("ArrowLeft")
    assert.equal(Number(await treeResizer.getAttribute("aria-valuenow")), treeWidth + 16)

    if (qaArtifacts !== undefined) {
      await page.emulateMedia({ colorScheme: "light" })
      await page.screenshot({ path: join(qaArtifacts, "workspace-files-light.png") })
      await page.emulateMedia({ colorScheme: "dark" })
      await page.screenshot({ path: join(qaArtifacts, "workspace-files-dark.png") })
      await page.emulateMedia({ colorScheme: "light" })
    }
  }

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
  const browserResizer = page.getByRole("separator", { name: "Resize workspace" })
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
  const browserResizeHandle = await page.getByRole("separator", { name: "Resize workspace" }).boundingBox()
  assert(browserResizeHandle, "the browser resize handle should be measurable")
  assert(
    Math.abs(browserResizeHandle.x + browserResizeHandle.width - browserViewport.x) <= 1,
    "the browser resize handle should end at the native viewport instead of overlapping it",
  )
  await browserResizer.hover({
    position: { x: browserResizeHandle.width / 2, y: browserResizeHandle.height - 24 },
  })
  await page.waitForFunction(() => {
    const element = document.querySelector<HTMLElement>('[role="separator"][aria-label="Resize workspace"]')
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

  await newBrowserTab(page)
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

  await newBrowserTab(page)
  const browserTabs = page.getByRole("tablist", { name: "Workspace tabs" }).locator('[role="tab"][data-workspace-tab-kind="browser"]')
  assert.equal(await browserTabs.count(), 2)
  assert.equal(
    await page.getByRole("textbox", { name: "Address and search" }).evaluate((element) => element === document.activeElement),
    true,
  )
  await navigate(page, fixtureUrl)
  const cookieGuest = await waitForPage(context, (candidate) => candidate.url() === `${fixtureUrl}/`)
  await cookieGuest.locator("#cookie").click()
  await newBrowserTab(page)
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
  await page.getByText("Downloaded", { exact: true }).waitFor()

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

  const closeLastBrowserTab = async (): Promise<void> => {
    await browserTabs.last().locator("..").locator('[aria-label^="Close "]').click()
  }
  while (await browserTabs.count() > 1) await closeLastBrowserTab()
  await closeLastBrowserTab()
  assert.equal(await browserTabs.count(), 0)
  await page.getByRole("textbox", { name: "Address and search" }).waitFor({ state: "hidden" })
  if (qaArtifacts !== undefined) await page.screenshot({ path: join(qaArtifacts, "blank-browser.png") })

  await newBrowserTab(page)
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
    if (electronProcess.exitCode === null && electronInspectorPort !== null) {
      await evaluateInElectronMain(
        electronInspectorPort,
        'setTimeout(() => process.getBuiltinModule("module").createRequire(process.cwd() + "/package.json")("electron").app.quit(), 0); true',
      ).catch(() => undefined)
    }
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
  await signalFixtureProcesses("TERM")
  await sleep(500)
  await signalFixtureProcesses("KILL")
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
