import { desktopAutomation as automation } from "../../../desktop/src/automation"
import { FileSystem } from "@effect/platform"
import { Context, Effect, Layer, Schema } from "effect"
import { _electron, type Page } from "playwright"
import { join } from "node:path"
import { AssertionFailure, InfrastructureFailure } from "./domain"

export const DesktopLaunch = Schema.Struct({ executable: Schema.String, profile: Schema.String, evidence: Schema.String,
  port: Schema.Int.pipe(Schema.between(1024, 65535)), environment: Schema.Record({ key: Schema.String, value: Schema.String }) })
export type DesktopLaunch = typeof DesktopLaunch.Type
export interface DesktopDriver {
  readonly navigate: (page: "discover" | "catalog" | "models" | "connections" | "usage" | "status" | "settings") => Effect.Effect<void, AssertionFailure>
  readonly host: () => Effect.Effect<string, AssertionFailure>
  readonly ready: () => Effect.Effect<void, AssertionFailure>
  readonly search: (modelId: string) => Effect.Effect<void, AssertionFailure>
  readonly download: (modelId: string) => Effect.Effect<void, AssertionFailure>
  readonly load: (modelId: string) => Effect.Effect<void, AssertionFailure>
  readonly connect: (harnessId: string) => Effect.Effect<void, AssertionFailure>
  readonly disconnect: (harnessId: string) => Effect.Effect<void, AssertionFailure>
  readonly theme: (theme: "light" | "dark" | "system") => Effect.Effect<void, AssertionFailure>
  readonly screenshot: (name: string) => Effect.Effect<string, AssertionFailure>
  readonly text: () => Effect.Effect<string, AssertionFailure>
  readonly chrome: () => Effect.Effect<void, AssertionFailure>
}
export const DesktopDriver = Context.GenericTag<DesktopDriver>("@magnitudedev/testing-lab/DesktopDriver")
// Playwright is the explicit Promise boundary. Test orchestration and lifecycle stay in Effect.
const action = <A>(description: string, run: () => Promise<A>) => Effect.tryPromise({ try: run,
  catch: error => new AssertionFailure({ message: `${description}: ${error instanceof Error ? error.message.slice(0, 1800) : "Playwright failed"}` }) })
export const playwrightDesktop = (config: DesktopLaunch, preparePage?: (page: Page) => Promise<void>) => Layer.scoped(DesktopDriver, Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  yield* fs.makeDirectory(config.profile, { recursive: true, mode: 0o700 })
  yield* fs.makeDirectory(config.evidence, { recursive: true, mode: 0o700 })
  let processLog = ""
  const collect = (chunk: Buffer) => { processLog = (processLog + chunk.toString("utf8")).slice(-2 * 1024 * 1024) }
  const app = yield* Effect.acquireRelease(Effect.tryPromise({ try: () => _electron.launch({ executablePath: config.executable, chromiumSandbox: true,
    env: { ...config.environment, MAGNITUDE_DEV_DATA_DIR: config.profile, MAGNITUDE_DEV_PORT: String(config.port), MAGNITUDE_SHELL_ENV_INHERITED: "1" }, timeout: 60_000 }),
    catch: error => new InfrastructureFailure({ operation: "desktop-launch", message: error instanceof Error ? error.message : "Packaged Electron launch failed" }) }),
  app => action("Close packaged application", () => app.close()).pipe(
    Effect.timeoutFail({ duration: "20 seconds", onTimeout: () => new AssertionFailure({ message: "Packaged application did not quit within 20 seconds" }) }),
    Effect.tapError(() => Effect.sync(() => { app.process().kill("SIGKILL") })),
    Effect.ensuring(fs.writeFileString(join(config.evidence, "desktop.log"), processLog.replace(/Bearer\s+[^\s"']+/gi, "Bearer [REDACTED]")).pipe(Effect.orDie)),
    Effect.orDie,
  ))
  app.process().stdout?.on("data", collect)
  app.process().stderr?.on("data", collect)
  const page = yield* action("Wait for packaged application window", () => app.firstWindow({ timeout: 60_000 }))
  page.setDefaultTimeout(30_000)
  if (preparePage) yield* action("Prepare UI resilience challenge", () => preparePage(page))
  yield* action("Start UI trace", () => app.context().tracing.start({ screenshots: true, snapshots: true, sources: false }))
  yield* Effect.addFinalizer(() => action("Save UI trace", () => app.context().tracing.stop({ path: join(config.evidence, "ui-trace.zip") })).pipe(Effect.timeout("20 seconds"), Effect.orDie))
  const navigate: DesktopDriver["navigate"] = name => action(`Open ${name}`, async () => {
    await page.getByTestId(automation.navigation(name)).click()
    await page.getByTestId(automation.page(name)).waitFor()
  })
  const card = (id: string) => page.getByTestId(automation.model(id))
  return {
    navigate,
    host: () => action("Verify packaged native host bridge", () => page.evaluate(async () => {
      const bridge = (window as unknown as { __magnitudeDesktop?: { applicationInfo: () => Promise<{ version: string }> } }).__magnitudeDesktop
      if (!bridge) throw new Error("Desktop preload did not expose its native host bridge")
      const info = await bridge.applicationInfo()
      if (!info.version || info.version === "unknown") throw new Error("Native host did not report the running application version")
      return info.version
    })).pipe(Effect.timeoutFail({ duration: "15 seconds", onTimeout: () => new AssertionFailure({ message: "Native host bridge did not respond within 15 seconds" }) })),
    ready: () => navigate("status").pipe(Effect.zipRight(action("Wait for packaged service readiness", () => page.getByTestId(automation.serviceReady).waitFor({ timeout: 180_000 })))),
    search: name => navigate("catalog").pipe(Effect.zipRight(action("Search model catalog", async () => {
      await page.getByTestId(automation.modelSearch).fill(name)
      await card(name).waitFor({ timeout: 120_000 })
    }))),
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
    screenshot: name => action("Capture UI evidence", async () => {
      if (!/^[a-z0-9-]+$/.test(name)) throw new Error("Invalid screenshot name")
      const path = join(config.evidence, `${name}.png`)
      await page.screenshot({ path, fullPage: true })
      return path
    }),
    text: () => action("Read visible application state", () => page.locator("body").innerText()),
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
