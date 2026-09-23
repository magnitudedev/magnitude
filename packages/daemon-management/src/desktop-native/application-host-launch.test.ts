import { chmod, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { dirname, join } from "node:path"
import { Effect, Fiber, Option, Schedule } from "effect"
import { afterEach, describe, expect, it, vi } from "vitest"
import { makeDesktopApplicationHost } from "./application-host"

afterEach(() => vi.unstubAllEnvs())

describe.skipIf(process.platform === "win32")("desktop application launch environment", () => {
  it("does not let an inherited ELECTRON_RUN_AS_NODE turn the desktop app into bare Node", async () => {
    const repository = await mkdtemp(join(tmpdir(), "magnitude-launch-"))
    try {
      const electron = join(repository, "node_modules/electron/dist", process.platform === "darwin" ? "Electron.app/Contents/MacOS/Electron" : "electron")
      const captured = join(repository, "environment.txt")
      await mkdir(dirname(electron), { recursive: true })
      await mkdir(join(repository, "desktop/out/main"), { recursive: true })
      await writeFile(join(repository, "desktop/out/main/main.js"), "")
      await writeFile(electron, `#!/bin/sh\n/usr/bin/env > "${captured}.partial" && mv "${captured}.partial" "${captured}"\n`)
      await chmod(electron, 0o755)
      vi.stubEnv("ELECTRON_RUN_AS_NODE", "1")
      vi.stubEnv("DISPLAY", ":0")
      vi.stubEnv("MAGNITUDE_DEV_DATA_DIR", repository)
      vi.stubEnv("MAGNITUDE_DESKTOP_STATE_DIR", join(repository, "state"))

      const host = makeDesktopApplicationHost(Option.some(repository))
      const environment = await Effect.runPromise(Effect.gen(function* () {
        const launch = yield* Effect.fork(host.startDesktopApplication)
        const output = yield* Effect.tryPromise(() => readFile(captured, "utf8")).pipe(
          Effect.retry({ schedule: Schedule.spaced("50 millis"), times: 100 }),
        )
        yield* Fiber.interrupt(launch)
        return output
      }))

      expect(environment).not.toMatch(/^ELECTRON_RUN_AS_NODE=/m)
      expect(environment).toMatch(/^MAGNITUDE_SHELL_ENV_INHERITED=1$/m)
      expect(process.env.ELECTRON_RUN_AS_NODE).toBe("1")
    } finally {
      await rm(repository, { recursive: true, force: true })
    }
  })
})
