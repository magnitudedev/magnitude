import { GuardedCommand, guardedCommandLayer } from "@magnitudedev/daemon-management/desktop-native"
import { Command } from "@effect/platform"
import { NodeContext } from "@effect/platform-node"
import { Effect, Fiber } from "effect"
import { chmod, mkdtemp, readFile, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { fileURLToPath } from "node:url"
import { join } from "node:path"
import { describe, expect, it } from "vitest"
import { harnessCommandExecutor, resolveHarnessEnvironment } from "./shell-env"

const fixture = async (source: string, test: (shell: string, root: string) => Promise<void>) => {
  const root = await mkdtemp(join(tmpdir(), "magnitude-shell-env-"))
  const shell = join(root, "shell")
  try { await writeFile(shell, `#!/bin/sh\n${source}\n`); await chmod(shell, 0o700); await test(shell, root) }
  finally { await rm(root, { recursive: true, force: true }) }
}
const guard = guardedCommandLayer(fileURLToPath(new URL(`../../packages/daemon-management/dist/native/${process.platform}-${process.arch}/magnitude-command`, import.meta.url)))
const run = <A, E>(effect: Effect.Effect<A, E, import("@effect/platform/CommandExecutor").CommandExecutor | GuardedCommand>) => Effect.runPromise(effect.pipe(Effect.provide([NodeContext.layer, guard])))
const alive = (pid: number) => { try { process.kill(pid, 0); return true } catch { return false } }

describe("harness environment", () => {
  it("resolves PATH and config without global mutations", () => fixture(
    'export PATH="/test/tools:/usr/bin:/bin"; export HERMES_HOME="/shell/hermes"; export MAGNITUDE_ENV_TEST="  spaced = value  "\nexec /bin/sh -c "$2"', async shell => {
      const before = { ...process.env }
      const result = await run(resolveHarnessEnvironment({ environment: { SHELL: shell, PATH: "/usr/bin:/bin", HERMES_HOME: "/explicit/hermes" } }))
      expect(result.PATH).toBe("/test/tools:/usr/bin:/bin")
      expect(result.HERMES_HOME).toBe("/explicit/hermes")
      expect(result.MAGNITUDE_ENV_TEST).toBe("  spaced = value  ")
      expect(result.ELECTRON_RUN_AS_NODE).toBeUndefined()
      expect(result.MAGNITUDE_RESOLVING_ENV).toBeUndefined()
      expect(process.env).toEqual(before)
    },
  ))
  it("falls back to login-only or inherited environment", () => fixture(
    '[ "$1" = "-ilc" ] && exit 3\nexport MAGNITUDE_ENV_TEST=fallback\nexec /bin/sh -c "$2"', async shell => {
      expect((await run(resolveHarnessEnvironment({ environment: { SHELL: shell } }))).MAGNITUDE_ENV_TEST).toBe("fallback")
      const missing = { SHELL: `${shell}-missing`, PATH: "/original" }
      expect(await run(resolveHarnessEnvironment({ environment: missing }))).toEqual(missing)
    },
  ))
  it("skips Windows, Nushell and inherited shell launches", () => fixture(
    'echo invoked > "$CHECK_FILE"; exit 1', async (shell, root) => {
      const environment = { SHELL: shell, CHECK_FILE: join(root, "called") }
      expect(await run(resolveHarnessEnvironment({ platform: "win32", environment }))).toEqual(environment)
      const inherited = { ...environment, MAGNITUDE_SHELL_ENV_INHERITED: "1" }
      expect(await run(resolveHarnessEnvironment({ environment: inherited }))).toEqual(inherited)
      const nu = { SHELL: "/missing/nu" }
      expect(await run(resolveHarnessEnvironment({ environment: nu }))).toEqual(nu)
      await expect(readFile(environment.CHECK_FILE)).rejects.toMatchObject({ code: "ENOENT" })
    },
  ))
  it("bounds hung probes and kills shells ignoring TERM", () => fixture(
    'echo $$ >> "$PID_FILE"\ntrap "" TERM\nwhile :; do /bin/sleep 30; done', async (shell, root) => {
      const environment = { SHELL: shell, PID_FILE: join(root, "pids") }
      const started = Date.now()
      expect(await run(resolveHarnessEnvironment({ environment, timeoutMilliseconds: 500 }))).toEqual(environment)
      expect(Date.now() - started).toBeLessThan(2500)
      const pids = (await readFile(environment.PID_FILE, "utf8")).trim().split("\n").map(Number)
      expect(pids).toHaveLength(2)
      expect(pids.some(alive)).toBe(false)
    },
  ))
  it("caps output and preserves command overrides", () => fixture(
    '/usr/bin/head -c 1100000 /dev/zero', async shell => {
      const environment = { SHELL: shell }
      expect(await run(resolveHarnessEnvironment({ environment }))).toEqual(environment)
      const output = await run(Effect.gen(function* () {
        const executor = yield* harnessCommandExecutor({ PATH: "/usr/bin:/bin", PI_CODING_AGENT_DIR: "/resolved", MAGNITUDE_ENV_TEST: "local" })
        return yield* executor.string(Command.make("/bin/sh", "-c", 'printf "%s:%s" "$PI_CODING_AGENT_DIR" "$MAGNITUDE_ENV_TEST"').pipe(Command.env({ PI_CODING_AGENT_DIR: "/command" })))
      }))
      expect(output).toBe("/command:local")
    },
  ))
  it("retires descendants holding output open after their shell exits", () => fixture(
    '/bin/sleep 30 &\necho $! >> "$PID_FILE"\nexit 0', async (shell, root) => {
      const environment = { SHELL: shell, PID_FILE: join(root, "pids") }
      expect(await run(resolveHarnessEnvironment({ environment, timeoutMilliseconds: 500 }))).toEqual(environment)
      const pids = (await readFile(environment.PID_FILE, "utf8")).trim().split("\n").map(Number)
      await expect.poll(() => pids.some(alive)).toBe(false)
    },
  ))
  it("cancels a pending probe with its owner", () => fixture(
    'echo $$ > "$PID_FILE"\ntrap "" TERM\nwhile :; do /bin/sleep 30; done', async (shell, root) => {
      const file = join(root, "pid")
      const fiber = Effect.runFork(resolveHarnessEnvironment({ environment: { SHELL: shell, PID_FILE: file } }).pipe(Effect.provide([NodeContext.layer, guard])))
      try { await expect.poll(async () => readFile(file, "utf8").then(Number).catch(() => 0)).toBeGreaterThan(0) }
      finally { await Effect.runPromise(Fiber.interrupt(fiber)) }
      expect(alive(Number(await readFile(file, "utf8")))).toBe(false)
    },
  ))
})
