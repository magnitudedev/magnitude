import { mkdtemp, readFile, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { ProcessGroupController } from "@magnitudedev/acn-protocol/coordination"
import { ProcessGroupControllerLive } from "@magnitudedev/acn-protocol/coordination/exact-process"
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

  it("remains observable after a pre-exit poll times out", async () => {
    await Effect.runPromise(
      Effect.scoped(
        Effect.gen(function* () {
          const spawned = yield* BunDetachedChildProcessSpawner.spawn([
            globalThis.process.execPath,
            "-e",
            "await Bun.sleep(50); process.stderr.write('candidate failed', () => process.exit(7))",
          ])
          const early = yield* spawned.exited.pipe(Effect.timeoutOption(Duration.millis(1)))
          expect(Option.isNone(early)).toBe(true)
          const exit = yield* spawned.exited.pipe(Effect.timeout(Duration.seconds(2)))
          expect(exit).toEqual({ code: 7, stderr: "candidate failed" })
          expect(yield* spawned.exited).toEqual(exit)
        }),
      ).pipe(Effect.provideService(ProcessGroupController, ProcessGroupControllerLive)),
    )
  })

  it("does not make scope closure wait for an admitted process stderr pipe", async () => {
    let admitted: Option.Option<Parameters<typeof ProcessGroupControllerLive.stop>[0]["leader"]> = Option.none()
    try {
      await Effect.runPromise(
        Effect.scoped(
          Effect.gen(function* () {
            const spawned = yield* BunDetachedChildProcessSpawner.spawn([
              globalThis.process.execPath,
              "-e",
              "process.stdin.resume(); await new Promise((resolve) => process.stdin.once('end', resolve)); setInterval(() => {}, 1000)",
            ])
            const identity = yield* ProcessGroupControllerLive.inspect(spawned.pid)
            if (Option.isNone(identity)) return yield* Effect.dieMessage("spawned candidate identity is absent")
            admitted = identity
            yield* spawned.confirmExactProcess(identity.value)
            yield* spawned.admit
          }),
        ).pipe(
          Effect.provideService(ProcessGroupController, ProcessGroupControllerLive),
          Effect.timeout(Duration.seconds(2)),
        ),
      )
    } finally {
      if (Option.isSome(admitted)) {
        await Effect.runPromise(ProcessGroupControllerLive.stop({ leader: admitted.value }))
      }
    }
  })

  it.skipIf(process.platform === "win32")(
    "observes and reaps an admitted root exit when a descendant inherited stderr",
    async () => {
      const root = await mkdtemp(join(tmpdir(), "magnitude-candidate-tree-"))
      const childPidPath = join(root, "child-pid")
      let childPid = 0
      try {
        await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
          const spawned = yield* BunDetachedChildProcessSpawner.spawn([
            process.execPath,
            "-e",
            "const child = Bun.spawn([process.execPath, '-e', 'setInterval(() => {}, 1000)'], { stdin: 'ignore', stdout: 'ignore', stderr: 'inherit' }); child.unref(); await Bun.write(process.argv.at(-1), String(child.pid)); process.stdin.resume(); await new Promise((resolve) => process.stdin.once('end', resolve)); process.stderr.write('candidate exited'); process.exit(9);",
            childPidPath,
          ])
          const identity = yield* ProcessGroupControllerLive.inspect(spawned.pid)
          if (Option.isNone(identity)) return yield* Effect.dieMessage("spawned candidate identity is absent")
          yield* spawned.confirmExactProcess(identity.value)
          while (!(yield* Effect.promise(() => Bun.file(childPidPath).exists()))) {
            yield* Effect.sleep(Duration.millis(5))
          }
          childPid = Number(yield* Effect.promise(() => readFile(childPidPath, "utf8")))
          yield* spawned.admit
          const exit = yield* spawned.exited.pipe(Effect.timeout(Duration.seconds(2)))
          expect(exit.code).toBe(9)
          expect(exit.stderr).toBe("candidate exited")
          yield* spawned.retireAdmittedGroup
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
