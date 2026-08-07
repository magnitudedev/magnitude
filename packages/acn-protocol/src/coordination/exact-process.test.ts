import { resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { Duration, Effect, Exit, Option } from "effect"
import { describe, expect, it } from "vitest"
import { ExactProcessControllerLive } from "./exact-process"
import type { ExactProcessInspectionFailed } from "./errors"
import type { ExactProcess } from "./schemas"

const waitForTreeAbsence = (
  process: ExactProcess,
): Effect.Effect<void, ExactProcessInspectionFailed> => Effect.gen(function* () {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    if (yield* ExactProcessControllerLive.treeAbsent(process)) return
    yield* Effect.sleep(Duration.millis(20))
  }
  return yield* Effect.dieMessage(`process tree ${process.pid} remained alive`)
})

describe("ExactProcessController", () => {
  it.skipIf(process.platform === "win32")(
    "rejects a stale identity and terminates one real process group",
    async () => {
      const fixture = resolve(fileURLToPath(new URL(".", import.meta.url)), "fixtures/process-tree.ts")
      const child = Bun.spawn([process.execPath, fixture], {
        detached: true,
        stdin: "ignore",
        stdout: "pipe",
        stderr: "inherit",
      })
      try {
        const reader = child.stdout.getReader()
        const first = await reader.read()
        reader.releaseLock()
        if (first.done) throw new Error("process-tree fixture exited before publishing its PID")
        const [pidText] = new TextDecoder().decode(first.value).trim().split(":")
        const pid = Number(pidText)
        const identity = Option.getOrThrow(await Effect.runPromise(
          ExactProcessControllerLive.inspect(pid),
        ))
        const exact = { pid, processStartIdentity: identity }

        expect(await Effect.runPromise(ExactProcessControllerLive.signal({
          ...exact,
          processStartIdentity: `${identity}-stale` as never,
        }, "term"))).toBe(false)
        expect(await Effect.runPromise(ExactProcessControllerLive.signalTree({
          ...exact,
          processStartIdentity: `${identity}-stale` as never,
        }, "term"))).toBe(false)
        expect(Option.contains(await Effect.runPromise(
          ExactProcessControllerLive.inspect(pid),
        ), identity)).toBe(true)

        expect(await Effect.runPromise(
          ExactProcessControllerLive.signalTree(exact, "term"),
        )).toBe(true)
        await child.exited
        await Effect.runPromise(waitForTreeAbsence(exact))
      } finally {
        try {
          child.kill(9)
        } catch {
          // The process group may already have completed and been reaped.
        }
        await child.exited
      }
    },
  )

  it.skipIf(process.platform !== "win32")(
    "fails closed when the recorded Windows root has exited",
    async () => {
      const child = Bun.spawn([process.execPath, "-e", "setInterval(() => {}, 1000)"], {
        stdin: "ignore",
        stdout: "ignore",
        stderr: "ignore",
      })
      const identity = Option.getOrThrow(await Effect.runPromise(
        ExactProcessControllerLive.inspect(child.pid),
      ))
      child.kill(9)
      await child.exited
      const exit = await Effect.runPromise(Effect.exit(
        ExactProcessControllerLive.treeAbsent({
          pid: child.pid,
          processStartIdentity: identity,
        }),
      ))
      expect(Exit.isFailure(exit)).toBe(true)
    },
  )

  it.skipIf(process.platform === "win32")(
    "can terminate a surviving process group after its recorded root exits",
    async () => {
      const fixture = resolve(fileURLToPath(new URL(".", import.meta.url)), "fixtures/process-tree.ts")
      const root = Bun.spawn([process.execPath, fixture], {
        detached: true,
        stdin: "ignore",
        stdout: "pipe",
        stderr: "inherit",
      })
      let exact: ExactProcess | undefined
      try {
        const reader = root.stdout.getReader()
        const first = await reader.read()
        reader.releaseLock()
        if (first.done) throw new Error("process-tree fixture exited before publishing its PID")
        const [pidText] = new TextDecoder().decode(first.value).trim().split(":")
        const pid = Number(pidText)
        const processStartIdentity = Option.getOrThrow(await Effect.runPromise(
          ExactProcessControllerLive.inspect(pid),
        ))
        exact = { pid, processStartIdentity }

        expect(await Effect.runPromise(
          ExactProcessControllerLive.signal(exact, "kill"),
        )).toBe(true)
        await root.exited
        expect(await Effect.runPromise(ExactProcessControllerLive.treeAbsent(exact))).toBe(false)
        expect(await Effect.runPromise(
          ExactProcessControllerLive.signalTree(exact, "kill"),
        )).toBe(true)
        await Effect.runPromise(waitForTreeAbsence(exact))
      } finally {
        if (exact !== undefined) {
          await Effect.runPromise(ExactProcessControllerLive.signalTree(exact, "kill")).catch(() => undefined)
        }
        try {
          root.kill(9)
        } catch {
          // The process group may already have completed and been reaped.
        }
        await root.exited
      }
    },
  )
})
