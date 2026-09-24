import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Deferred, Effect, Fiber, Option } from "effect"
import { join } from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"
import { BunSqliteDriverLayer } from "../bun"
import { runHeadlessApplication } from "./headless-application"
import { nativeHostLayer } from "./index"
import { MacUpdateAdmission, nativeMacUpdateAdmission } from "./mac-update-lease"

const addon = fileURLToPath(new URL(`../../dist/native/darwin-${process.arch}/desktop-host.node`, import.meta.url))

describe.skipIf(process.platform !== "darwin")("Installed macOS headless admission", () => {
  it("retains shared installation admission before update initialization and releases it on cancellation", () => Effect.runPromise(
    Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const admission = yield* MacUpdateAdmission
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-headless-admission-" })
      const bundle = join(root, "Magnitude.app")
      const resourcesDirectory = join(bundle, "Contents/Resources")
      yield* fs.makeDirectory(resourcesDirectory, { recursive: true })
      yield* fs.copyFile(addon, join(resourcesDirectory, "desktop-host.node"))
      yield* fs.writeFileString(join(bundle, "Contents/Info.plist"),
        '<?xml version="1.0"?><plist version="1.0"><dict><key>CFBundleIdentifier</key><string>dev.magnitude.admission-fixture</string></dict></plist>')
      const initialized = yield* Deferred.make<void>()
      const options = {
        runtime: { _tag: "Installed" as const, resourcesDirectory },
        profile: { dataDirectory: join(root, "data"), isolated: true, port: 11101, endpoint: "http://127.0.0.1:11101" },
        stateDirectory: join(root, "state"), home: root, environment: {}, stop: Effect.never,
        observe: () => Effect.void,
        initializeUpdates: Deferred.succeed(initialized, undefined).pipe(Effect.zipRight(Effect.never)),
      }
      yield* Effect.scoped(Effect.gen(function* () {
        expect(Option.isSome(yield* admission.exclusive(bundle))).toBe(true)
        expect(yield* runHeadlessApplication(options).pipe(Effect.isFailure)).toBe(true)
        expect(yield* Deferred.isDone(initialized)).toBe(false)
      }))
      const owner = yield* runHeadlessApplication(options).pipe(Effect.forkScoped)
      yield* Deferred.await(initialized).pipe(Effect.timeout("10 seconds"))
      expect(Option.isNone(yield* admission.exclusive(bundle))).toBe(true)
      yield* Fiber.interrupt(owner)
      expect(Option.isSome(yield* admission.exclusive(bundle))).toBe(true)
    })).pipe(Effect.provide([nativeHostLayer(addon), nativeMacUpdateAdmission(addon), BunContext.layer, BunSqliteDriverLayer]))
  ))
})
