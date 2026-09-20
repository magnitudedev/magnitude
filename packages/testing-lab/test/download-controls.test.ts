import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Schema } from "effect"
import { dirname, join, resolve } from "node:path"
import { createRequire } from "node:module"
import { _electron } from "playwright"
import { expect, test } from "vitest"
import { playwrightDownloads } from "../src/download-controls"
import { challengePresentation } from "../test-support/presentation-challenge"

test("download controls survive rewording and distinguish transfer, failure, retry and completion", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-download-controls-" })
  const html = join(root, "fixture.html"), main = join(root, "main.cjs"), model = "model:gguf:q4"
  yield* fs.writeFileString(html, `<!doctype html><body>
    <article data-testid="desktop.model.${model}" data-acquisition-state="NotInstalled" data-model-installed="false">
      <button data-testid="desktop.model-download" onclick="const card=this.parentElement;card.querySelector('[role=alert]').hidden=true;setTimeout(()=>{card.dataset.acquisitionState='Installing'},20)">Download</button>
      <div data-testid="desktop.model-download-progress" data-download-stage="downloading" data-download-completed-bytes="16" data-download-total-bytes="100"></div>
      <p role="alert" hidden>Network unavailable</p>
    </article></body>`)
  yield* fs.writeFileString(main, `const {app,BrowserWindow}=require('electron');app.whenReady().then(()=>{const window=new BrowserWindow({show:false});window.loadFile(${yield* Schema.encode(Schema.parseJson(Schema.String))(html)})});app.on('window-all-closed',()=>app.quit());`)
  const executablePath = resolve(dirname(createRequire(import.meta.url).resolve("electron/package.json")), "dist",
    process.platform === "darwin" ? "Electron.app/Contents/MacOS/Electron" : process.platform === "win32" ? "electron.exe" : "electron")
  const app = yield* Effect.acquireRelease(Effect.tryPromise(() => _electron.launch({ executablePath, args: [main] })), app => Effect.promise(() => app.close()).pipe(Effect.interruptible, Effect.timeout("5 seconds"), Effect.orDie))
  const page = yield* Effect.promise(() => app.firstWindow())
  yield* Effect.promise(() => challengePresentation(page))
  const controls = playwrightDownloads(page)
  yield* controls.absent(model)
  yield* controls.begin(model)
  expect(yield* controls.transferring(model)).toEqual({ completedBytes: 16, totalBytes: 100 })
  yield* Effect.promise(() => page.getByTestId(`desktop.model.${model}`).evaluate(element => {
    element.setAttribute("data-acquisition-state", "InstallFailed")
    element.querySelector<HTMLElement>('[role="alert"]')!.hidden = false
  }))
  yield* controls.failed(model)
  expect((yield* controls.complete(model).pipe(Effect.either))._tag).toBe("Left")
  yield* controls.begin(model)
  expect(yield* controls.transferring(model)).toEqual({ completedBytes: 16, totalBytes: 100 })
  yield* Effect.promise(() => page.getByTestId(`desktop.model.${model}`).evaluate(element => {
    element.setAttribute("data-acquisition-state", "Installed")
    element.setAttribute("data-model-installed", "true")
  }))
  yield* controls.complete(model)
  expect((yield* controls.transferring(model).pipe(Effect.either))._tag).toBe("Left")
  expect((yield* controls.failed(model).pipe(Effect.either))._tag).toBe("Left")
  yield* Effect.promise(() => page.getByTestId(`desktop.model.${model}`).evaluate(element => {
    element.setAttribute("data-acquisition-state", "NotInstalled")
    element.querySelector<HTMLButtonElement>('[data-testid="desktop.model-download"]')!.onclick = () => {
      element.setAttribute("data-acquisition-state", "InstallFailed")
      element.querySelector<HTMLElement>('[role="alert"]')!.hidden = false
    }
  }))
  const failedStart = yield* controls.begin(model).pipe(Effect.either)
  expect(failedStart._tag).toBe("Left")
  if (failedStart._tag === "Left") expect(failedStart.left.message).toContain("Network unavailable")
})).pipe(Effect.provide(BunContext.layer))), 20000)
