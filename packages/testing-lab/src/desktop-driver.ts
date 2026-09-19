import { playwrightUpdates, type DesktopUpdates } from "./update-controls"
import { ApplicationIdentity, ReadyApplicationSnapshot } from "./application-identity"
import { desktopAutomation as automation } from "../../../desktop/src/automation"
import { FileSystem } from "@effect/platform"
import { Cause, Context, Effect, Layer, Schema } from "effect"
import { _electron, type Page } from "playwright"
import type { ChildProcess } from "node:child_process"
import { join } from "node:path"
import { AssertionFailure, InfrastructureFailure } from "./domain"

export const DesktopLaunch = Schema.Struct({ executable: Schema.String, profile: Schema.String, evidence: Schema.String,
  port: Schema.Int.pipe(Schema.between(1024, 65535)), environment: Schema.Record({ key: Schema.String, value: Schema.String }) })
export type DesktopLaunch = typeof DesktopLaunch.Type
export interface DesktopDriver {
  readonly updates: DesktopUpdates
  readonly navigate: (page: "discover" | "catalog" | "models" | "connections" | "usage" | "status" | "settings") => Effect.Effect<void, AssertionFailure>
  readonly identity: () => Effect.Effect<ApplicationIdentity, AssertionFailure>
  readonly host: () => Effect.Effect<string, AssertionFailure>
  readonly serviceFailure: () => Effect.Effect<string, AssertionFailure>
  readonly ready: () => Effect.Effect<void, AssertionFailure>
  readonly search: (modelId: string) => Effect.Effect<void, AssertionFailure>
  readonly details: (modelId: string) => Effect.Effect<void, AssertionFailure>
  readonly download: (modelId: string) => Effect.Effect<void, AssertionFailure>
  readonly load: (modelId: string) => Effect.Effect<void, AssertionFailure>
  readonly connect: (harnessId: string) => Effect.Effect<void, AssertionFailure>
  readonly connectionFailure: (harness: string, fileName: string) => Effect.Effect<string, AssertionFailure>
  readonly disconnect: (harnessId: string) => Effect.Effect<void, AssertionFailure>
  readonly theme: (theme: "light" | "dark" | "system") => Effect.Effect<void, AssertionFailure>
  readonly verifyTheme: (theme: "light" | "dark" | "system") => Effect.Effect<void, AssertionFailure>
  readonly screenshot: (name: string) => Effect.Effect<string, AssertionFailure>
  readonly text: () => Effect.Effect<string, AssertionFailure>
  readonly quit: () => Effect.Effect<void, AssertionFailure>
  readonly restartForUpdate: () => Effect.Effect<void, AssertionFailure>
  readonly chrome: () => Effect.Effect<void, AssertionFailure>
}
export const DesktopDriver = Context.GenericTag<DesktopDriver>("@magnitudedev/testing-lab/DesktopDriver")
// Playwright is the explicit Promise boundary. Test orchestration and lifecycle stay in Effect.
const action = <A>(description: string, run: () => Promise<A>) => Effect.tryPromise({ try: run,
  catch: error => new AssertionFailure({ message: `${description}: ${error instanceof Error ? error.message.slice(0, 1800) : "Playwright failed"}` }) })
export const playwrightDesktop = (config: DesktopLaunch, preparePage?: (page: Page) => Promise<void>, onCleanupError?: (detail: string) => void) => Layer.scoped(DesktopDriver, Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  yield* fs.makeDirectory(config.profile, { recursive: true, mode: 0o700 })
  yield* fs.makeDirectory(config.evidence, { recursive: true, mode: 0o700 })
  // With an explicit result collector, keep cleanup failures separate from the test failure.
  const reportCleanup = <A, E>(effect: Effect.Effect<A, E>) => effect.pipe(Effect.catchAllCause(cause =>
    onCleanupError ? Effect.sync(() => onCleanupError(Cause.pretty(cause))) : Effect.die(cause)))
  let nativeProcess: ChildProcess | undefined
  let processLog = ""
  const collect = (chunk: Buffer) => { processLog = (processLog + chunk.toString("utf8")).slice(-2 * 1024 * 1024) }
  const app = yield* Effect.acquireRelease(Effect.tryPromise({ try: () => _electron.launch({ executablePath: config.executable, chromiumSandbox: true,
    env: { ...config.environment, MAGNITUDE_DEV_DATA_DIR: config.profile, MAGNITUDE_DEV_PORT: String(config.port), MAGNITUDE_SHELL_ENV_INHERITED: "1" }, timeout: 60_000 }),
    catch: error => new InfrastructureFailure({ operation: "desktop-launch", message: error instanceof Error ? error.message : "Packaged Electron launch failed" }) }),
  app => action("Close packaged application", async () => {
    if (!nativeProcess || (nativeProcess.exitCode === null && nativeProcess.signalCode === null)) await app.close()
  }).pipe(
    Effect.interruptible, Effect.timeoutFail({ duration: "20 seconds", onTimeout: () => new AssertionFailure({ message: "Packaged application did not quit within 20 seconds" }) }),
    Effect.tapError(() => Effect.sync(() => {
      const child = nativeProcess
      // Playwright launches a separate process group on Unix; reap its helpers as well.
      if (child?.pid && child.exitCode === null && child.signalCode === null) {
        try { if (process.platform === "win32") child.kill("SIGKILL"); else process.kill(-child.pid, "SIGKILL") }
        catch (error) { if (!(error instanceof Error && "code" in error && error.code === "ESRCH")) throw error }
      }
    })),
    Effect.ensuring(fs.writeFileString(join(config.evidence, "desktop.log"), processLog.replace(/Bearer\s+[^\s"']+/gi, "Bearer [REDACTED]")).pipe(Effect.orDie)),
    reportCleanup,
  ))
  nativeProcess = app.process()
  nativeProcess.stdout?.on("data", collect)
  nativeProcess.stderr?.on("data", collect)
  const page = yield* action("Wait for packaged application window", () => app.firstWindow({ timeout: 60_000 }))
  page.setDefaultTimeout(30_000)
  if (preparePage) yield* action("Prepare UI resilience challenge", () => preparePage(page))
  yield* action("Start UI trace", () => app.context().tracing.start({ screenshots: true, snapshots: true, sources: false }))
  const saveTrace = yield* Effect.cached(action("Save UI trace", () => app.context().tracing.stop({ path: join(config.evidence, "ui-trace.zip") })).pipe(
    Effect.timeoutFail({ duration: "20 seconds", onTimeout: () => new AssertionFailure({ message: "UI trace did not finish" }) })))
  yield* Effect.addFinalizer(() => saveTrace.pipe(Effect.interruptible, reportCleanup))
  const navigate: DesktopDriver["navigate"] = name => action(`Open ${name}`, async () => {
    await page.getByTestId(automation.navigation(name)).click()
    await page.getByTestId(automation.page(name)).waitFor()
  })
  const card = (id: string) => page.getByTestId(automation.model(id))
  const updates = playwrightUpdates(page, navigate("settings"))
  const exited = Effect.async<void>(resume => {
    const child = nativeProcess!
    if (child.exitCode !== null || child.signalCode !== null) { resume(Effect.void); return }
    const done = () => resume(Effect.void)
    child.once("exit", done)
    return Effect.sync(() => { child.removeListener("exit", done) })
  }).pipe(Effect.flatMap(() => nativeProcess!.exitCode === 0 && nativeProcess!.signalCode === null
    ? Effect.void : new AssertionFailure({ message: "Application did not exit cleanly through normal quit" })))
  return {
    updates,
    // Finalize diagnostics before native replacement retires the Playwright connection.
    // The caller separately observes the replacement owner; this proves only the old process exit.
    restartForUpdate: () => saveTrace.pipe(Effect.zipRight(updates.action("restart")), Effect.zipRight(exited),
      Effect.timeoutFail({ duration: "2 minutes", onTimeout: () => new AssertionFailure({ message: "Application update did not retire the previous process" }) })),
    navigate,
    identity: () => Effect.gen(function* () {
      const wire = yield* action("Read native service ownership", () => page.evaluate(() => new Promise<unknown>((resolve, reject) => {
        const bridge = (window as unknown as { __magnitudeDesktop?: { observe: (value: (snapshot: unknown) => void, error: (message: string) => void) => () => void } }).__magnitudeDesktop
        if (!bridge) { reject(new Error("Missing native observation bridge")); return }
        let unsubscribe: (() => void) | undefined
        let finished = false
        const finish = () => { finished = true; clearTimeout(timer); unsubscribe?.() }
        const timer = setTimeout(() => { finish(); reject(new Error("Native ownership observation timed out")) }, 15_000)
        unsubscribe = bridge.observe(value => { finish(); resolve(value) }, message => { finish(); reject(new Error(message)) })
        if (finished) unsubscribe()
      })))
      const snapshot = yield* Schema.decodeUnknown(ReadyApplicationSnapshot)(wire).pipe(Effect.mapError(() => new AssertionFailure({ message: "Native service observation is not ready or lacks identity" })))
      if (snapshot.pid !== nativeProcess!.pid || snapshot.service.health.pid === snapshot.pid || snapshot.endpoint !== `http://127.0.0.1:${config.port}`) return yield* new AssertionFailure({ message: "Native service identity does not belong to the isolated application" })
      return ApplicationIdentity.make({ applicationPid: snapshot.pid, servicePid: snapshot.service.health.pid, serviceInstance: snapshot.service.health.id })
    }),
    host: () => action("Verify packaged native host bridge", () => page.evaluate(async () => {
      const bridge = (window as unknown as { __magnitudeDesktop?: { applicationInfo: () => Promise<{ version: string }> } }).__magnitudeDesktop
      if (!bridge) throw new Error("Desktop preload did not expose its native host bridge")
      const info = await bridge.applicationInfo()
      if (!info.version || info.version === "unknown") throw new Error("Native host did not report the running application version")
      return info.version
    })).pipe(Effect.timeoutFail({ duration: "15 seconds", onTimeout: () => new AssertionFailure({ message: "Native host bridge did not respond within 15 seconds" }) })),
    serviceFailure: () => navigate("status").pipe(Effect.zipRight(action("Observe failed service startup", async () => {
      const status = page.getByTestId(automation.page("status"))
      const alert = status.getByRole("alert").first()
      await alert.waitFor({ timeout: 180_000 })
      await status.getByTestId(automation.serviceReady).waitFor({ state: "hidden" })
      const message = await alert.innerText()
      if (!message.trim()) throw new Error("Service failure displayed an empty diagnostic")
      return message
    }))),
    ready: () => navigate("status").pipe(Effect.zipRight(action("Wait for packaged service readiness", async () => {
      const status = page.getByTestId(automation.page("status"))
      const ready = status.getByTestId(automation.serviceReady)
      const failure = status.getByRole("alert").first()
      await ready.or(failure).first().waitFor({ timeout: 180_000 })
      if (!await ready.isVisible()) throw new Error(await failure.innerText())
    }))),
    search: name => navigate("catalog").pipe(Effect.zipRight(action("Search model catalog", async () => {
      await page.getByTestId(automation.modelSearch).fill(name)
      await card(name).waitFor({ timeout: 120_000 })
    }))),
    details: name => action("Open model details", async () => {
      // The disclosure's accessibility relationship identifies its content independently of copy.
      const disclosure = card(name).locator('button[aria-controls][aria-expanded]')
      const controlled = await disclosure.getAttribute("aria-controls")
      if (!controlled) throw new Error("Model details has no controlled content identity")
      if (await disclosure.getAttribute("aria-expanded") !== "true") await disclosure.click()
      await disclosure.and(page.locator('[aria-expanded="true"]')).waitFor()
      await page.locator(`[id="${controlled.replaceAll("\\", "\\\\").replaceAll('"', '\\"')}"]`).waitFor()
    }),
    download: name => action("Download model through packaged UI", async () => {
      const model = card(name)
      await model.getByTestId(automation.modelDownload).click({ timeout: 120_000 })
      const complete = model.and(page.locator('[data-model-installed="true"]'))
      const failure = model.getByRole("alert")
      await model.getByTestId(automation.modelDownloadProgress).or(complete).or(failure).first().waitFor({ timeout: 60_000 })
      await complete.or(failure).first().waitFor({ timeout: 30 * 60_000 })
      if (await failure.count()) throw new Error(await failure.allTextContents().then(messages => messages.join("; ")))
      await complete.waitFor()
    }),
    load: name => action("Load model through packaged UI", async () => {
      const model = card(name)
      await model.getByTestId(automation.modelLoad).click()
      const loaded = model.and(page.locator('[data-model-ready="true"]'))
      const failure = model.getByRole("alert")
      await loaded.or(failure).first().waitFor({ timeout: 5 * 60_000 })
      if (await failure.count()) throw new Error(await failure.allTextContents().then(messages => messages.join("; ")))
      await loaded.waitFor()
    }),
    connect: name => navigate("connections").pipe(Effect.zipRight(action(`Connect ${name} through the app`, async () => {
      const harness = page.getByTestId(automation.harness(name))
      await harness.getByTestId(automation.harnessConnect).click()
      await harness.and(page.locator('[data-connected="true"]')).waitFor()
      await harness.getByTestId(automation.harnessConnect).and(page.locator(':enabled')).waitFor()
    }))),
    connectionFailure: (name, fileName) => navigate("connections").pipe(Effect.zipRight(action(`Observe ${name} configuration error`, async () => {
      await page.getByTestId(automation.harness(name)).getByTestId(automation.harnessConnect).click()
      // Alert and repair guidance may be separate UI elements; only the affected file is stable.
      const alert = page.getByTestId(automation.page("connections")).getByRole("alert").first()
      const guidance = page.getByTestId(automation.harness(name)).getByText(fileName, { exact: false })
      await alert.waitFor()
      await guidance.waitFor()
      const message = await alert.innerText()
      if (!message.trim()) throw new Error("Connection failure displayed an empty alert")
      return `${message}\n${await guidance.innerText()}`
    }))),
    disconnect: name => navigate("connections").pipe(Effect.zipRight(action(`Disconnect ${name} through the app`, async () => {
      const harness = page.getByTestId(automation.harness(name))
      await harness.getByTestId(automation.harnessDisconnect).click()
      await harness.and(page.locator('[data-connected="false"]')).waitFor()
    }))),
    theme: theme => navigate("settings").pipe(Effect.zipRight(action("Change appearance", async () => {
      await page.getByTestId(automation.theme(theme)).click()
      await page.getByTestId(automation.theme(theme)).and(page.locator('[aria-pressed="true"]')).waitFor()
    }))),
    verifyTheme: theme => navigate("settings").pipe(Effect.zipRight(action("Verify saved appearance", () =>
      page.getByTestId(automation.theme(theme)).and(page.locator('[aria-pressed="true"]')).waitFor()))),
    screenshot: name => action("Capture UI evidence", async () => {
      if (!/^[a-z0-9-]+$/.test(name)) throw new Error("Invalid screenshot name")
      const path = join(config.evidence, `${name}.png`)
      await page.screenshot({ path, fullPage: true })
      return path
    }),
    text: () => action("Read visible application state", () => page.locator("body").innerText()),
    quit: () => Effect.gen(function* () {
      yield* saveTrace
      // Playwright's Electron close handler invokes app.quit(), allowing the application's
      // normal before-quit shutdown path to run. Only cleanup may force termination.
      yield* action("Quit packaged application", () => app.close())
      yield* exited
    }).pipe(Effect.timeoutFail({ duration: "30 seconds", onTimeout: () => new AssertionFailure({ message: "Application quit did not terminate the process within 30 seconds" }) })),
    chrome: () => action("Exercise packaged window controls", async () => {
      const toggle = page.getByTestId(automation.sidebarToggle)
      const sidebar = page.getByTestId(automation.sidebar)
      await toggle.click()
      await toggle.and(page.locator('[aria-expanded="false"]')).waitFor()
      await sidebar.and(page.locator('[aria-hidden="true"]')).waitFor({ state: "attached" })
      await toggle.click()
      await toggle.and(page.locator('[aria-expanded="true"]')).waitFor()
      await sidebar.and(page.locator('[aria-hidden="false"]')).waitFor()
      await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0]!.minimize())
      await app.evaluate(async ({ BrowserWindow }) => { const w = BrowserWindow.getAllWindows()[0]!; if (!w.isMinimized()) await new Promise<void>(resolve => w.once("minimize", () => resolve())); w.restore() })
      await app.evaluate(async ({ BrowserWindow }) => {
        const window = BrowserWindow.getAllWindows()[0]!
        await new Promise<void>((resolve, reject) => {
          const complete = () => { clearTimeout(timeout); resolve() }
          const timeout = setTimeout(() => { window.removeListener("hide", complete); reject(new Error("Closing the window did not retain a hidden window")) }, 10_000)
          window.once("hide", complete)
          window.close()
          if (!window.isVisible()) { window.removeListener("hide", complete); complete() }
        })
      })
      await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0]!.show())
    }),
  } satisfies DesktopDriver
}))
