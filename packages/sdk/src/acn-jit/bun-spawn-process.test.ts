import { mkdtemp, readFile, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { ProcessGroupController, ProcessGroupControllerLive } from "@magnitudedev/acn-protocol/coordination"
import { Duration, Effect, Option } from "effect"
import { describe, expect, it } from "vitest"
import { BunDetachedChildProcessSpawner } from "./bun-spawn-process"

describe("BunDetachedChildProcessSpawner", () => {
  it("returns a scope-owned candidate with a mandatory PID", async () => {
    await Effect.runPromise(
      Effect.scoped(
        Effect.gen(function* () {
          const spawned = yield* BunDetachedChildProcessSpawner.spawn([
            globalThis.process.execPath,
            "-e",
            [
              `process.stderr.write("x".repeat(${70 * 1024}))`,
              `process.stderr.write("\\ncandidate failed\\n", () => process.exit(7))`,
            ].join(";"),
          ])
          expect(spawned.pid).toBeGreaterThan(0)
          const exit = yield* spawned.exited
          expect(exit.code).toBe(7)
          expect(new TextEncoder().encode(exit.stderr).length).toBeLessThanOrEqual(64 * 1024)
          expect(exit.stderr).toMatch(/candidate failed$/)
        }),
      ).pipe(Effect.provideService(ProcessGroupController, ProcessGroupControllerLive)),
    )
  })

  it.skipIf(process.platform === "win32")(
    "reaps the candidate group when the root exits before its descendant",
    async () => {
      const root = await mkdtemp(join(tmpdir(), "magnitude-candidate-tree-"))
      const childPidPath = join(root, "child-pid")
      let childPid = 0
      try {
        await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
          const spawned = yield* BunDetachedChildProcessSpawner.spawn([
            process.execPath,
            "-e",
            "const child = Bun.spawn([process.execPath, '-e', 'setInterval(() => {}, 1000)'], { stdin: 'ignore', stdout: 'ignore', stderr: 'ignore' }); child.unref(); await Bun.write(process.argv.at(-1), String(child.pid));",
            childPidPath,
          ])
          const identity = yield* ProcessGroupControllerLive.inspect(spawned.pid)
          if (Option.isNone(identity)) return yield* Effect.dieMessage("spawned candidate identity is absent")
          yield* spawned.confirmExactProcess(identity.value)
          while (!(yield* Effect.promise(() => Bun.file(childPidPath).exists()))) {
            yield* Effect.sleep(Duration.millis(5))
          }
          childPid = Number(yield* Effect.promise(() => readFile(childPidPath, "utf8")))
          yield* spawned.exited
        }).pipe(Effect.provideService(ProcessGroupController, ProcessGroupControllerLive))))
        expect(Option.isNone(await Effect.runPromise(
          ProcessGroupControllerLive.inspect(childPid),
        ))).toBe(true)
      } finally {
        if (childPid > 0) {
          try { process.kill(childPid, "SIGKILL") } catch { /* already reaped */ }
        }
        await rm(root, { recursive: true, force: true })
      }
    },
  )
})
