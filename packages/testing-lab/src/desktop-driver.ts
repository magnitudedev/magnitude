import { FileSystem } from "@effect/platform"
import { Context, Effect, Layer, Schema } from "effect"
import { _electron } from "playwright"
import { join } from "node:path"
import { AssertionFailure, InfrastructureFailure } from "./domain"

export const DesktopLaunch = Schema.Struct({ executable: Schema.String, profile: Schema.String, evidence: Schema.String,
  port: Schema.Int.pipe(Schema.between(1024, 65535)), environment: Schema.Record({ key: Schema.String, value: Schema.String }) })
export type DesktopLaunch = typeof DesktopLaunch.Type
export interface DesktopDriver {
  readonly navigate: (page: "Discover" | "Catalog" | "My Models" | "Connections" | "Usage" | "Status" | "Settings") => Effect.Effect<void, AssertionFailure>
  readonly host: () => Effect.Effect<string, AssertionFailure>
  readonly ready: () => Effect.Effect<void, AssertionFailure>
  readonly search: (modelName: string) => Effect.Effect<void, AssertionFailure>
  readonly download: (modelName: string) => Effect.Effect<void, AssertionFailure>
  readonly load: (modelName: string) => Effect.Effect<void, AssertionFailure>
  readonly connect: (harnessName: string) => Effect.Effect<void, AssertionFailure>
  readonly disconnect: (harnessName: string) => Effect.Effect<void, AssertionFailure>
  readonly theme: (theme: "Light" | "Dark" | "System") => Effect.Effect<void, AssertionFailure>
  readonly screenshot: (name: string) => Effect.Effect<string, AssertionFailure>
  readonly text: () => Effect.Effect<string, AssertionFailure>
  readonly chrome: () => Effect.Effect<void, AssertionFailure>
}
export const DesktopDriver = Context.GenericTag<DesktopDriver>("@magnitudedev/testing-lab/DesktopDriver")
// Playwright is the explicit Promise boundary. Test orchestration and lifecycle stay in Effect.
const action = <A>(description: string, run: () => Promise<A>) => Effect.tryPromise({ try: run,
  catch: error => new AssertionFailure({ message: `${description}: ${error instanceof Error ? error.message.slice(0, 1800) : "Playwright failed"}` }) })
export const playwrightDesktop = (config: DesktopLaunch) => Layer.scoped(DesktopDriver, Effect.gen(function* () {
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
  yield* action("Start UI trace", () => app.context().tracing.start({ screenshots: true, snapshots: true, sources: false }))
  yield* Effect.addFinalizer(() => action("Save UI trace", () => app.context().tracing.stop({ path: join(config.evidence, "ui-trace.zip") })).pipe(Effect.timeout("20 seconds"), Effect.orDie))
  const navigate: DesktopDriver["navigate"] = name => action(`Open ${name}`, async () => {
    await page.getByRole("button", { name, exact: true }).click()
    await page.getByRole("heading", { name, exact: true }).waitFor()
  })
  const card = (name: string) => page.locator("article").filter({ has: page.getByRole("heading", { name, exact: true }) })
  return {
    navigate,
    host: () => action("Verify packaged native host bridge", () => page.evaluate(async () => {
      const bridge = (window as unknown as { __magnitudeDesktop?: { applicationInfo: () => Promise<{ version: string }> } }).__magnitudeDesktop
      if (!bridge) throw new Error("Desktop preload did not expose its native host bridge")
      const info = await bridge.applicationInfo()
      if (!info.version || info.version === "unknown") throw new Error("Native host did not report the running application version")
      return info.version
    })).pipe(Effect.timeoutFail({ duration: "15 seconds", onTimeout: () => new AssertionFailure({ message: "Native host bridge did not respond within 15 seconds" }) })),
    ready: () => navigate("Status").pipe(Effect.zipRight(action("Wait for packaged service readiness", () => page.getByLabel("Service ready", { exact: true }).waitFor({ timeout: 180_000 })))),
    search: name => navigate("Catalog").pipe(Effect.zipRight(action("Search model catalog", async () => {
      await page.getByRole("textbox", { name: "Search models", exact: true }).fill(name)
      await page.getByRole("heading", { name, exact: true }).waitFor({ timeout: 120_000 })
    }))),
    download: name => action("Download model through packaged UI", async () => {
      const model = card(name)
      await model.getByRole("button", { name: /^Download \(/ }).click({ timeout: 120_000 })
      const complete = model.getByRole("button", { name: "Load model", exact: true })
      const failure = model.getByRole("alert")
      await model.getByRole("progressbar", { name: "Download progress" }).or(complete).or(failure).first().waitFor({ timeout: 60_000 })
      await complete.or(failure).first().waitFor({ timeout: 30 * 60_000 })
      if (await failure.count()) throw new Error(await failure.allTextContents().then(messages => messages.join("; ")))
      await complete.waitFor()
    }),
    load: name => action("Load model through packaged UI", async () => {
      const model = card(name)
      await model.getByRole("button", { name: "Load model", exact: true }).click()
      const loaded = model.getByText("Loaded", { exact: true })
      const failure = model.getByRole("alert")
      await loaded.or(failure).first().waitFor({ timeout: 5 * 60_000 })
      if (await failure.count()) throw new Error(await failure.allTextContents().then(messages => messages.join("; ")))
      await loaded.waitFor()
    }),
    connect: name => navigate("Connections").pipe(Effect.zipRight(action(`Connect ${name} through the app`, async () => {
      const harness = page.getByRole("article", { name, exact: true })
      await harness.getByRole("button", { name: "Connect", exact: true }).click()
      await harness.getByText("Connected", { exact: true }).waitFor()
    }))),
    disconnect: name => navigate("Connections").pipe(Effect.zipRight(action(`Disconnect ${name} through the app`, async () => {
      const harness = page.getByRole("article", { name, exact: true })
      await harness.getByRole("button", { name: "Disconnect", exact: true }).click()
      await harness.getByText("Not connected", { exact: true }).waitFor()
    }))),
    theme: theme => navigate("Settings").pipe(Effect.zipRight(action("Change appearance", async () => {
      await page.getByRole("group", { name: "Theme", exact: true }).getByRole("button", { name: theme, exact: true }).click()
      await page.getByRole("group", { name: "Theme", exact: true }).locator('[aria-pressed="true"]').filter({ hasText: theme }).waitFor()
    }))),
    screenshot: name => action("Capture UI evidence", async () => {
      if (!/^[a-z0-9-]+$/.test(name)) throw new Error("Invalid screenshot name")
      const path = join(config.evidence, `${name}.png`)
      await page.screenshot({ path, fullPage: true })
      return path
    }),
    text: () => action("Read visible application state", () => page.locator("body").innerText()),
    chrome: () => action("Exercise packaged window controls", async () => {
      await page.getByRole("button", { name: "Collapse sidebar", exact: true }).click()
      await page.getByRole("button", { name: "Expand sidebar", exact: true }).waitFor()
      await page.waitForFunction(() => document.querySelector("aside")?.getBoundingClientRect().width === 0)
      await page.getByRole("button", { name: "Expand sidebar", exact: true }).click()
      await page.waitForFunction(() => (document.querySelector("aside")?.getBoundingClientRect().width ?? 0) > 0)
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
