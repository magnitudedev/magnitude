import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Context, Effect, Layer, Option, Schema } from "effect"
import { createRequire } from "node:module"
import { dirname, join } from "node:path"
import { expect, it } from "vitest"
import { DesktopDriver, playwrightDesktop } from "../src/desktop-driver"

const quote = (value: string) => `'${value.replaceAll("'", "'\\''")}'`

it.skipIf(process.platform === "win32").each([0, 7, null])("observes the exact update-retiring process with exit %s", async exit => {
  const cleanup: string[] = []
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-update-handoff-" })
    const executable = join(root, "electron-fixture"), main = join(root, "main.cjs")
    const html = join(root, "fixture.html")
    const electron = join(dirname(createRequire(import.meta.url).resolve("electron/package.json")), "dist",
      process.platform === "darwin" ? "Electron.app/Contents/MacOS/Electron" : "electron")
    // Test composition only: ordinary Electron controls its real child lifetime.
    yield* fs.writeFileString(executable, `#!/bin/sh\nexec ${quote(electron)} "$@" ${quote(main)}\n`, { mode: 0o700 })
    yield* fs.writeFileString(html, `<!doctype html><body><button data-testid="desktop.navigate.settings">Settings</button>
      <section data-testid="desktop.page.settings"><button data-testid="desktop.update.restart" onclick="require('electron').ipcRenderer.send('restart')">Restart</button></section></body>`)
    yield* fs.writeFileString(main, `const {app,BrowserWindow,ipcMain}=require('electron');
      app.whenReady().then(()=>{const w=new BrowserWindow({show:false,webPreferences:{nodeIntegration:true,contextIsolation:false}});w.loadFile(${yield* Schema.encode(Schema.parseJson(Schema.String))(html)});});
      ipcMain.on('restart',()=>{${exit === null ? "" : `setTimeout(()=>${exit === 0 ? "app.quit()" : `app.exit(${exit})`},50);`}});
      app.on('window-all-closed',()=>app.quit());`)
    const driver = Context.get(yield* Layer.build(playwrightDesktop({ executable, profile: join(root, "profile"),
      evidence: join(root, "evidence"), port: 11459, environment: { HOME: root, PATH: process.env.PATH ?? "" },
    }, undefined, detail => { cleanup.push(detail) })), DesktopDriver)
    if (exit === null) {
      expect(Option.isNone(yield* driver.restartForUpdate().pipe(Effect.timeoutOption("300 millis")))).toBe(true)
      // Cancelling the observation must not terminate the still-owned application.
      yield* driver.screenshot("still-running")
      yield* driver.quit()
    } else {
      const result = yield* driver.restartForUpdate().pipe(Effect.either)
      expect(result._tag).toBe(exit === 0 ? "Right" : "Left")
      if (result._tag === "Left") expect(result.left.message).toContain("did not exit cleanly")
    }
    expect(Number((yield* fs.stat(join(root, "evidence/ui-trace.zip"))).size)).toBeGreaterThan(0)
  })).pipe(Effect.provide(BunContext.layer)))
  expect(cleanup).toEqual([])
}, 20_000)
