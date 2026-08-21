import { resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { Duration, Effect, Exit, Option } from "effect"
import { describe, expect, it } from "vitest"
import { ProcessGroupControllerLive } from "./exact-process"
import type { ProcessGroupObservationFailed } from "./errors"
import type { ProcessGroup } from "./schemas"

const waitForGroupAbsence = (
  group: ProcessGroup,
): Effect.Effect<void, ProcessGroupObservationFailed> => Effect.gen(function* () {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    if ((yield* ProcessGroupControllerLive.observeGroup(group))._tag === "ProcessGroupAbsent") return
    yield* Effect.sleep(Duration.millis(20))
  }
  return yield* Effect.dieMessage(`process group ${group.leader.pid} remained alive`)
})

describe("exact process capabilities", () => {
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
          ProcessGroupControllerLive.inspect(pid),
        ))
        const exact = { pid, processStartIdentity: identity }
        const group = { leader: exact }

        expect((await Effect.runPromise(ProcessGroupControllerLive.signalGroup({
          leader: { ...exact, processStartIdentity: `${identity}-stale` as never },
        }, "term")))._tag).toBe("ProcessGroupLeaderChanged")
        expect(Option.contains(await Effect.runPromise(
          ProcessGroupControllerLive.inspect(pid),
        ), identity)).toBe(true)

        expect((await Effect.runPromise(
          ProcessGroupControllerLive.signalGroup(group, "term"),
        ))._tag).toBe("ProcessGroupSignaled")
        await child.exited
        await Effect.runPromise(waitForGroupAbsence(group))
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
        ProcessGroupControllerLive.inspect(child.pid),
      ))
      child.kill(9)
      await child.exited
      const exit = await Effect.runPromise(Effect.exit(
        ProcessGroupControllerLive.observeGroup({
          leader: { pid: child.pid, processStartIdentity: identity },
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
      let group: ProcessGroup | undefined
      try {
        const reader = root.stdout.getReader()
        const first = await reader.read()
        reader.releaseLock()
        if (first.done) throw new Error("process-tree fixture exited before publishing its PID")
        const [pidText] = new TextDecoder().decode(first.value).trim().split(":")
        const pid = Number(pidText)
        const processStartIdentity = Option.getOrThrow(await Effect.runPromise(
          ProcessGroupControllerLive.inspect(pid),
        ))
        group = { leader: { pid, processStartIdentity } }

        expect(await Effect.runPromise(
          Effect.sync(() => root.kill(9)),
        ))
        await root.exited
        expect((await Effect.runPromise(ProcessGroupControllerLive.observeGroup(group)))._tag)
          .toBe("ProcessGroupPresent")
        expect((await Effect.runPromise(
          ProcessGroupControllerLive.signalGroup(group, "kill"),
        ))._tag).toBe("ProcessGroupSignaled")
        await Effect.runPromise(waitForGroupAbsence(group))
      } finally {
        if (group !== undefined) {
          await Effect.runPromise(ProcessGroupControllerLive.signalGroup(group, "kill")).catch(() => undefined)
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
