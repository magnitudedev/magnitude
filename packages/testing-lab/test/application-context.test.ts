import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect } from "effect"
import { join } from "node:path"
import { expect, test } from "vitest"
import { prepareApplicationContext } from "../src/application-context"
import { DisposableDesktopUser } from "../src/desktop-environment"

test("isolated consumers share one endpoint, profile and harness home with scoped native control", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-context-" })
  const context = yield* Effect.scoped(prepareApplicationContext(root, 11489, { HOME: "/ambient", PATH: "/tools" }))
  expect(context.mode).toBe("isolated")
  expect(context.environment.MAGNITUDE_DEV_DATA_DIR).toBe(context.profile)
  expect(context.environment.MAGNITUDE_DEV_PORT).toBe(String(context.port))
  expect(context.harnessHome).toBe(join(context.profile, "harness-home"))
  expect(context.environment.HOME).toBe(join(root, "home"))
  expect(context.environment.PATH).toBe("/tools")
  if (process.platform !== "win32") expect(yield* fs.exists(context.environment.MAGNITUDE_DESKTOP_STATE_DIR!)).toBe(false)
})).pipe(Effect.provide(BunContext.layer))))

test("installed context preserves the qualified user environment and rejects existing data including dangling links", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-user-context-" }), home = join(root, "user")
  yield* fs.makeDirectory(home)
  const environment = { HOME: home, USERPROFILE: home, APPDATA: join(home, "actual-roaming"), LOCALAPPDATA: join(home, "actual-local") }
  const prepare = prepareApplicationContext(join(root, "work"), 11489, environment).pipe(Effect.provideService(DisposableDesktopUser, { home }))
  const context = yield* prepare
  expect(context.mode).toBe("installed-user")
  expect(context.port).toBe(10100)
  expect(context.profile).toBe(join(home, ".magnitude"))
  expect(context.harnessHome).toBe(home)
  expect(context.environment).toEqual({ ...environment, MAGNITUDE_SHELL_ENV_INHERITED: "1" })
  expect(yield* fs.exists(context.profile)).toBe(false)
  yield* fs.makeDirectory(context.profile)
  expect((yield* prepare.pipe(Effect.either))._tag).toBe("Left")
  yield* fs.remove(context.profile, { recursive: true })
  if (process.platform !== "win32") {
    yield* fs.symlink(join(root, "absent"), context.profile)
    expect((yield* prepare.pipe(Effect.either))._tag).toBe("Left")
  }
})).pipe(Effect.provide(BunContext.layer))))
