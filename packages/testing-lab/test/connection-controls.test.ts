import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Schema } from "effect"
import { dirname, join, resolve } from "node:path"
import { createRequire } from "node:module"
import { _electron } from "playwright"
import { expect, test } from "vitest"
import { observeConnectionFailure } from "../src/desktop-driver"
import { challengePresentation } from "../test-support/presentation-challenge"

test("connection errors reveal collapsed file guidance despite presentation changes", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-connection-controls-" })
  const html = join(root, "fixture.html"), main = join(root, "main.cjs")
  yield* fs.writeFileString(html, `<!doctype html><body>
    <section data-testid="desktop.page.connections">
      <p role="alert" hidden>Configuration cannot be read; repair the listed file.</p>
      <article data-testid="desktop.harness.opencode">
        <button data-testid="desktop.harness-connect" onclick="document.querySelector('[role=alert]').hidden=false">Connect</button>
        <details><summary>A differently worded disclosure</summary><ul><li>/isolated/.config/opencode/opencode.json</li></ul></details>
      </article>
    </section></body>`)
  yield* fs.writeFileString(main, `const {app,BrowserWindow}=require('electron');app.whenReady().then(()=>{const w=new BrowserWindow({show:false});w.loadFile(${yield* Schema.encode(Schema.parseJson(Schema.String))(html)})});app.on('window-all-closed',()=>app.quit());`)
  const executablePath = resolve(dirname(createRequire(import.meta.url).resolve("electron/package.json")), "dist",
    process.platform === "darwin" ? "Electron.app/Contents/MacOS/Electron" : process.platform === "win32" ? "electron.exe" : "electron")
  const app = yield* Effect.acquireRelease(Effect.tryPromise(() => _electron.launch({ executablePath, args: [main] })),
    app => Effect.promise(() => app.close()).pipe(Effect.interruptible, Effect.timeout("5 seconds"), Effect.orDie))
  const page = yield* Effect.promise(() => app.firstWindow())
  yield* Effect.promise(async () => { page.setDefaultTimeout(2000); await challengePresentation(page) })
  expect(yield* Effect.promise(() => page.getByText("/isolated/.config/opencode/opencode.json", { exact: true }).isVisible())).toBe(false)
  for (const attempt of ["collapsed", "already open"] as const) {
    const message = yield* observeConnectionFailure(page, "opencode", "opencode.json")
    expect(message, attempt).toContain("Configuration cannot be read")
    expect(message, attempt).toContain("/isolated/.config/opencode/opencode.json")
    expect(yield* Effect.promise(() => page.getByText("/isolated/.config/opencode/opencode.json", { exact: true }).isVisible())).toBe(true)
  }
  yield* Effect.promise(() => page.getByRole("alert").evaluate(element => { element.textContent = "" }))
  expect((yield* observeConnectionFailure(page, "opencode", "opencode.json").pipe(Effect.either))._tag).toBe("Left")
})).pipe(Effect.provide(BunContext.layer))), 20_000)
