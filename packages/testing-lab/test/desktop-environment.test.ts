import { Effect } from "effect"
import { join } from "node:path"
import { expect, test } from "vitest"
import { desktopEnvironment, DisposableDesktopUser } from "../src/desktop-environment"

test("isolated app and CLI environments must already agree on profile and endpoint", () => Effect.runPromise(Effect.gen(function* () {
  const config = { mode: "isolated" as const, profile: "/lab/profile", port: 11479,
    environment: { MAGNITUDE_DEV_DATA_DIR: "/lab/profile", MAGNITUDE_DEV_PORT: "11479", HOME: "/lab/home" } }
  expect((yield* desktopEnvironment(config)).MAGNITUDE_DEV_DATA_DIR).toBe(config.environment.MAGNITUDE_DEV_DATA_DIR)
  for (const environment of [{}, { ...config.environment, MAGNITUDE_DEV_PORT: "10100" }, { ...config.environment, MAGNITUDE_DEV_DATA_DIR: "/other" }]) {
    expect((yield* desktopEnvironment({ ...config, environment }).pipe(Effect.either))._tag).toBe("Left")
  }
})))

test("installed mode requires guest-user authority and preserves normal product paths", () => Effect.runPromise(Effect.gen(function* () {
  const home = process.platform === "win32" ? "C:\\Users\\labworker" : "/home/labworker"
  const config = { mode: "installed-user" as const, profile: join(home, ".magnitude"), port: 10100, environment: { HOME: home, USERPROFILE: home } }
  expect((yield* desktopEnvironment(config).pipe(Effect.either))._tag).toBe("Left")
  const run = (value: Parameters<typeof desktopEnvironment>[0]) => desktopEnvironment(value).pipe(Effect.provideService(DisposableDesktopUser, { home }))
  const environment = yield* run(config)
  expect(environment).toEqual({ ...config.environment, MAGNITUDE_SHELL_ENV_INHERITED: "1" })
  for (const override of ["MAGNITUDE_DEV_DATA_DIR", "MAGNITUDE_DEV_PORT", "MAGNITUDE_DESKTOP_STATE_DIR"]) {
    expect((yield* run({ ...config, environment: { ...config.environment, [override]: "" } }).pipe(Effect.either))._tag).toBe("Left")
  }
  for (const value of [{ ...config, port: 11479 }, { ...config, profile: join(home, "another-profile") },
    { ...config, environment: { ...config.environment, HOME: "/other" } }]) {
    expect((yield* run(value).pipe(Effect.either))._tag).toBe("Left")
  }
})))
