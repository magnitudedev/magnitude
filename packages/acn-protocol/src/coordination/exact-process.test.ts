import { resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { Effect, Exit, Option } from "effect"
import { describe, expect, it } from "vitest"
import { ProcessGroupControllerLive } from "./exact-process"
import type { ProcessGroup } from "./schemas"

const spawnProcessTree = () => {
  const fixture = resolve(fileURLToPath(new URL(".", import.meta.url)), "fixtures/process-tree.ts")
  return Bun.spawn([process.execPath, fixture], {
    detached: true,
    stdin: "ignore",
    stdout: "pipe",
    stderr: "inherit",
  })
}

const publishedRootPid = async (child: ReturnType<typeof spawnProcessTree>): Promise<number> => {
  const reader = child.stdout.getReader()
  const first = await reader.read()
  reader.releaseLock()
  if (first.done) throw new Error("process-tree fixture exited before publishing its PID")
  const [pidText] = new TextDecoder().decode(first.value).trim().split(":")
  return Number(pidText)
}

const observedTag = (group: ProcessGroup) =>
  Effect.runPromise(ProcessGroupControllerLive.observe(group)).then((observed) => observed._tag)

describe("ProcessGroupController", () => {
  it.skipIf(process.platform === "win32")(
    "refuses a replaced leader and stops one real process group",
    async () => {
      const child = spawnProcessTree()
      try {
        const pid = await publishedRootPid(child)
        const leader = Option.getOrThrow(await Effect.runPromise(ProcessGroupControllerLive.inspect(pid)))
        const group: ProcessGroup = { leader }
        expect(await observedTag(group)).toBe("ProcessGroupLeaderLive")

        const stale: ProcessGroup = {
          leader: { pid, processStartIdentity: `${leader.processStartIdentity}-stale` as never },
        }
        expect(await observedTag(stale)).toBe("ProcessGroupLeaderReplaced")
        expect((await Effect.runPromise(ProcessGroupControllerLive.stop(stale)))._tag)
          .toBe("ProcessGroupLeaderReplaced")
        expect(await observedTag(group)).toBe("ProcessGroupLeaderLive")

        expect((await Effect.runPromise(ProcessGroupControllerLive.stop(group)))._tag)
          .toBe("ProcessGroupStopped")
        await child.exited
        expect(await observedTag(group)).toBe("ProcessGroupAbsent")
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
      const leader = Option.getOrThrow(await Effect.runPromise(
        ProcessGroupControllerLive.inspect(child.pid),
      ))
      child.kill(9)
      await child.exited
      const exit = await Effect.runPromise(Effect.exit(ProcessGroupControllerLive.observe({ leader })))
      expect(Exit.isFailure(exit)).toBe(true)
    },
  )

  it.skipIf(process.platform === "win32")(
    "reports survivors after the recorded root exits and stops them",
    async () => {
      const root = spawnProcessTree()
      let group: ProcessGroup | undefined
      try {
        const pid = await publishedRootPid(root)
        const leader = Option.getOrThrow(await Effect.runPromise(ProcessGroupControllerLive.inspect(pid)))
        group = { leader }

        root.kill(9)
        await root.exited
        expect(await observedTag(group)).toBe("ProcessGroupSurvivorsOnly")
        expect((await Effect.runPromise(ProcessGroupControllerLive.stop(group)))._tag)
          .toBe("ProcessGroupStopped")
        expect(await observedTag(group)).toBe("ProcessGroupAbsent")
      } finally {
        if (group !== undefined) {
          await Effect.runPromise(ProcessGroupControllerLive.stop(group)).catch(() => undefined)
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
