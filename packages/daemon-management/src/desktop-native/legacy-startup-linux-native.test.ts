import { randomUUID } from "node:crypto"
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises"
import { homedir, tmpdir } from "node:os"
import { join } from "node:path"
import { Effect, Option } from "effect"
import { describe, expect, it } from "vitest"
import { LegacyStartupCommands, NativeLegacyStartupCommands } from "./legacy-startup-command"
import { makeLinuxLegacyStartup } from "./legacy-startup-linux"

// Explicit opt-in: this acceptance test registers disposable units in a real user manager.
describe.skipIf(process.platform !== "linux" || process.env.MAGNITUDE_TEST_SYSTEMD !== "1")("native systemd migration", () => {
  it.each(["enabled", "enabled-runtime", "disabled"])("retires a real %s user service and preserves its preference", preference =>
    Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const commands = yield* LegacyStartupCommands
      const unit = `magnitude-acceptance-${process.pid}-${randomUUID()}.service`
      const path = join(homedir(), ".config/systemd/user", unit)
      const directory = yield* Effect.acquireRelease(
        Effect.promise(() => mkdtemp(join(tmpdir(), "magnitude-systemd-"))),
        root => Effect.gen(function* () {
          yield* commands.run("systemctl", ["--user", "disable", "--runtime", unit]).pipe(Effect.ignore)
          yield* commands.run("systemctl", ["--user", "disable", "--now", unit]).pipe(Effect.ignore)
          yield* Effect.promise(() => rm(path, { force: true }))
          yield* commands.run("systemctl", ["--user", "daemon-reload"]).pipe(Effect.ignore)
          yield* Effect.promise(() => rm(root, { recursive: true, force: true }))
        }),
      )
      const executable = join(directory, "magnitude-service")
      yield* Effect.promise(async () => {
        await mkdir(join(homedir(), ".config/systemd/user"), { recursive: true })
        await writeFile(executable, "#!/bin/sh\nexec /bin/sleep 120\n", { mode: 0o700 })
        await writeFile(path, `[Unit]\nDescription=Magnitude local inference service\nAfter=network.target\n\n[Service]\nType=simple\nExecStart="${executable}" "serve"\nRestart=on-failure\nRestartSec=2\n\n[Install]\nWantedBy=default.target\n`, { flag: "wx", mode: 0o600 })
      })
      const command = (args: readonly string[]) => commands.run("systemctl", ["--user", ...args]).pipe(Effect.tap(result => Effect.sync(() => {
        expect(result.code, result.stderr).toBe(0)
      })))
      yield* command(["daemon-reload"])
      if (preference !== "disabled") yield* command(["enable", ...(preference === "enabled-runtime" ? ["--runtime"] : []), unit])
      yield* command(["start", unit])
      const adapter = yield* makeLinuxLegacyStartup({ home: homedir(), unit })
      const snapshot = Option.getOrThrow(yield* adapter.inspect)
      expect(snapshot.enabled).toBe(preference === "enabled")
      const pid = Option.getOrThrow(snapshot.runningPid)
      expect(() => process.kill(pid, 0)).not.toThrow()
      yield* adapter.unregister(snapshot)
      expect(() => process.kill(pid, 0)).toThrow()
      expect(Option.isNone(yield* adapter.inspect)).toBe(true)
      yield* adapter.unregister(snapshot)
      expect(Option.isNone(yield* adapter.inspect)).toBe(true)
    }).pipe(Effect.provide(NativeLegacyStartupCommands)))), 30_000)
})
