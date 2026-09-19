import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Option, Schema } from "effect"
import { dirname, join, resolve } from "node:path"
import { createRequire } from "node:module"
import { _electron } from "playwright"
import { expect, test } from "vitest"
import { playwrightUpdates } from "../src/update-controls"
import { challengePresentation } from "../test-support/presentation-challenge"

test("update controls survive presentation changes and stop on a rendered failure", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-update-controls-" })
  const html = join(root, "fixture.html"), main = join(root, "main.cjs")
  yield* fs.writeFileString(html, `<!doctype html><body>
    <section style="display:flex;flex-direction:column-reverse" data-testid="desktop.updates" data-update-state="Idle" data-update-check="Idle">
      <input type="checkbox" data-testid="desktop.update.automatic" onchange="this.dataset.changes=String(Number(this.dataset.changes??0)+1);const next=this.checked;this.checked=!next;setTimeout(()=>{this.checked=next},50)">
      <button data-testid="desktop.update.check" onclick="this.parentElement.dataset.updateState='Available';this.parentElement.dataset.updateVersion='0.1.4'">Check</button>
      <button data-testid="desktop.update.download" onclick="this.parentElement.dataset.updateState='Ready'">Download</button>
      <button data-testid="desktop.update.restart" onclick="this.parentElement.dataset.updateState='Closed'">Restart</button>
      <button data-testid="desktop.update.discard" onclick="this.parentElement.dataset.updateState='Idle';delete this.parentElement.dataset.updateVersion">Discard</button>
    </section></body>`)
  yield* fs.writeFileString(main, `const { app, BrowserWindow } = require('electron'); app.whenReady().then(() => {
    const window = new BrowserWindow({show: false}); window.loadFile(${yield* Schema.encode(Schema.parseJson(Schema.String))(html)});
  }); app.on('window-all-closed', () => app.quit());`)
  const executablePath = resolve(dirname(createRequire(import.meta.url).resolve("electron/package.json")), "dist",
    process.platform === "darwin" ? "Electron.app/Contents/MacOS/Electron" : process.platform === "win32" ? "electron.exe" : "electron")
  const app = yield* Effect.acquireRelease(Effect.tryPromise(() => _electron.launch({ executablePath, args: [main] })),
    app => Effect.promise(() => app.close()).pipe(Effect.interruptible, Effect.timeout("5 seconds"), Effect.orDie))
  const page = yield* Effect.promise(() => app.firstWindow())
  yield* Effect.promise(() => challengePresentation(page))
  const controls = playwrightUpdates(page, Effect.void)
  yield* controls.automatic(true)
  expect(yield* Effect.promise(() => page.getByTestId("desktop.update.automatic").isChecked())).toBe(true)
  yield* controls.automatic(true)
  yield* controls.automatic(false)
  expect(yield* Effect.promise(() => page.getByTestId("desktop.update.automatic").getAttribute("data-changes"))).toBe("2")
  yield* controls.action("check")
  expect(Option.getOrThrow((yield* controls.wait("Available")).version)).toBe("0.1.4")
  yield* controls.action("download")
  expect((yield* controls.wait("Ready")).state).toBe("Ready")
  yield* controls.action("restart")
  yield* controls.wait("Closed")
  yield* controls.action("discard")
  expect(Option.isNone((yield* controls.wait("Idle")).version)).toBe(true)
  yield* Effect.promise(() => page.getByTestId("desktop.updates").evaluate(element => element.setAttribute("data-update-state", "Failed")))
  const failed = yield* controls.wait("Ready").pipe(Effect.either)
  expect(failed._tag).toBe("Left")
}).pipe(Effect.timeout("20 seconds"))).pipe(Effect.provide(BunContext.layer))), 30_000)
