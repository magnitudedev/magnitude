import { execFileSync } from "node:child_process"
import { resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { Effect, Exit, Option } from "effect"
import { describe, expect, it, vi, type MockInstance } from "vitest"
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
  it.skipIf(process.platform !== "darwin")(
    "preserves C-locale process identities across caller locales",
    async () => {
      const started = execFileSync("/bin/ps", ["-o", "lstart=", "-p", String(process.pid)], {
        encoding: "utf8",
        env: { ...process.env, LC_ALL: "C" },
      }).trim()
      const boot = execFileSync("/usr/sbin/sysctl", ["-n", "kern.bootsessionuuid"], {
        encoding: "utf8",
      }).trim().toLowerCase()
      const expected = `darwin:${boot}:${started}`
      const modulePath = fileURLToPath(new URL("./exact-process.ts", import.meta.url))
      const script = `
        import { Effect, Option } from "effect"
        import { ProcessGroupControllerLive } from ${JSON.stringify(modulePath)}
        const observed = await Effect.runPromise(ProcessGroupControllerLive.inspect(${process.pid}))
        console.log(Option.getOrThrow(observed).processStartIdentity)
      `
      for (const locale of [
        { LANG: "C", LC_TIME: "C", LC_ALL: "C" },
        { LANG: "en_US.UTF-8", LC_TIME: "en_GB.UTF-8", LC_ALL: "" },
        { LANG: "fr_FR.UTF-8", LC_TIME: "", LC_ALL: "" },
        { LANG: "C", LC_TIME: "C", LC_ALL: "en_GB.UTF-8" },
      ]) {
        const child = Bun.spawn([process.execPath, "--eval", script], {
          env: { ...process.env, ...locale },
          stdin: "ignore",
          stdout: "pipe",
          stderr: "pipe",
        })
        try {
          const [stdout, stderr, code] = await Promise.all([
            new Response(child.stdout).text(),
            new Response(child.stderr).text(),
            child.exited,
          ])
          expect(code, stderr).toBe(0)
          expect(stdout.trim(), JSON.stringify(locale)).toBe(expected)
        } finally {
          child.kill()
          await child.exited
        }
      }
    },
  )

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

  it.skipIf(process.platform === "win32")("proves group absence after a permission error racing child exit", async () => {
    const child = spawnProcessTree()
    const originalKill = process.kill.bind(process)
    let group: ProcessGroup | undefined
    let injected = false
    let spy: MockInstance<typeof process.kill> | undefined
    try {
      const pid = await publishedRootPid(child)
      group = { leader: Option.getOrThrow(await Effect.runPromise(ProcessGroupControllerLive.inspect(pid))) }
      spy = vi.spyOn(process, "kill").mockImplementation((target, signal) => {
        if (target === -pid && signal === "SIGTERM" && !injected) {
          injected = true
          originalKill(target, signal)
          throw Object.assign(new Error("kill EPERM"), { code: "EPERM" })
        }
        return originalKill(target, signal)
      })
      expect((await Effect.runPromise(ProcessGroupControllerLive.stop(group)))._tag).toBe("ProcessGroupStopped")
      expect(injected).toBe(true)
      await child.exited
      expect(await observedTag(group)).toBe("ProcessGroupAbsent")
    } finally {
      spy?.mockRestore()
      if (group) await Effect.runPromise(ProcessGroupControllerLive.stop(group))
      try { child.kill(9) } catch {}
      await child.exited
    }
  })

  it.skipIf(process.platform === "win32")("preserves permission failure when the group remains present", async () => {
    const child = spawnProcessTree()
    const originalKill = process.kill.bind(process)
    let group: ProcessGroup | undefined
    let spy: MockInstance<typeof process.kill> | undefined
    try {
      const pid = await publishedRootPid(child)
      group = { leader: Option.getOrThrow(await Effect.runPromise(ProcessGroupControllerLive.inspect(pid))) }
      spy = vi.spyOn(process, "kill").mockImplementation((target, signal) => {
        if (target === -pid && signal === "SIGTERM") throw Object.assign(new Error("kill EPERM"), { code: "EPERM" })
        return originalKill(target, signal)
      })
      const result = await Effect.runPromise(ProcessGroupControllerLive.stop(group, { termWait: "30 millis", killWait: "30 millis" }).pipe(Effect.either))
      expect(result._tag).toBe("Left")
      if (result._tag === "Left") expect(result.left._tag).toBe("ProcessGroupSignalPermissionDenied")
      expect(await observedTag(group)).toBe("ProcessGroupLeaderLive")
    } finally {
      spy?.mockRestore()
      if (group) await Effect.runPromise(ProcessGroupControllerLive.stop(group))
      try { child.kill(9) } catch {}
      await child.exited
    }
  })

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
