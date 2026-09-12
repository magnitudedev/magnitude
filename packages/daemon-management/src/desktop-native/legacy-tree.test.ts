import { BunContext } from "@effect/platform-bun"
import { Effect, Layer, Option } from "effect"
import { describe, expect, it } from "vitest"
import { ProcessGroupController, ProcessStartIdentitySchema } from "@magnitudedev/utils/process-groups"
import { ProcessGroupControllerLive } from "@magnitudedev/utils/process-groups/native"
import { LegacyOwner } from "./legacy-owner"
import { captureLegacyTree, LegacyProcessTable, selectLegacyGroups, UnixLegacyProcessTable } from "./legacy-tree"

const owner = LegacyOwner.make({ pid: 100, processStartIdentity: ProcessStartIdentitySchema.make("root"), port: 1234 })
const rows = [
  { pid: 100, parent: 1, group: 100 },
  { pid: 101, parent: 100, group: 101 },
  { pid: 102, parent: 101, group: 101 },
  { pid: 999, parent: 1, group: 999 },
]
const run = Effect.runPromise
const forbidden = Effect.dieMessage("Capture must not mutate processes")
const processes = ProcessGroupController.of({
  inspect: pid => Effect.succeed(Option.some({ pid, processStartIdentity: ProcessStartIdentitySchema.make(pid === 100 ? "root" : "engine") })),
  currentProcess: forbidden, observe: () => forbidden, stop: () => forbidden, waitForGroupExit: () => forbidden,
})
const capture = (table: typeof LegacyProcessTable.Service, process = processes) => captureLegacyTree(owner).pipe(
  Effect.provideService(LegacyProcessTable, table), Effect.provideService(ProcessGroupController, process))

describe("legacy process-tree capture", () => {
  it("captures the service and separately grouped inference descendants only", async () => {
    expect(await run(selectLegacyGroups(owner, rows))).toEqual([100, 101])
    const tree = await run(capture({ read: Effect.succeed(rows) }))
    expect(tree.groups.map(group => group.leader.pid)).toEqual([100, 101])
  })
  it("rejects a root that does not lead its group", async () => {
    await expect(run(selectLegacyGroups(owner, [{ pid: 100, parent: 1, group: 99 }]))).rejects.toThrow("does not lead")
  })
  it("never claims an unrelated group through one descendant", async () => {
    await expect(run(selectLegacyGroups(owner, [rows[0]!, { pid: 101, parent: 100, group: 999 }, rows[3]!]))).rejects.toThrow("unverified process group")
  })
  it("rejects a group containing an unrelated member", async () => {
    await expect(run(selectLegacyGroups(owner, [...rows, { pid: 500, parent: 1, group: 101 }]))).rejects.toThrow("unrelated process")
  })
  it("refuses stale owner identity before reading the process table", async () => {
    await expect(run(capture({ read: forbidden }, { ...processes, inspect: () => Effect.succeed(Option.none()) }))).rejects.toThrow("identity changed or exited")
  })
  it("fails when the engine group changes during capture", async () => {
    let reads = 0
    await expect(run(capture({ read: Effect.sync(() => reads++ === 0 ? rows : rows.filter(row => row.pid !== 101 && row.pid !== 102)) }))).rejects.toThrow("groups changed")
  })
  it("fails on group PID reuse even when ancestry still looks the same", async () => {
    let engineReads = 0
    await expect(run(capture({ read: Effect.succeed(rows) }, { ...processes, inspect: pid => Effect.succeed(Option.some({
      pid, processStartIdentity: ProcessStartIdentitySchema.make(pid === 100 ? "root" : engineReads++ === 0 ? "engine" : "replacement"),
    })) }))).rejects.toThrow("group identity changed")
  })

  it.skipIf(process.platform === "win32")("captures real isolated service and inference groups", async () => {
    const childCode = "Bun.spawn([process.execPath, '-e', 'setInterval(() => {}, 1000)'], {stdin:'ignore',stdout:'ignore',stderr:'ignore'});setInterval(() => {}, 1000)"
    const code = `const child = Bun.spawn([process.execPath, '-e', ${JSON.stringify(childCode)}], {detached:true,stdin:'ignore',stdout:'ignore',stderr:'ignore'});console.log(child.pid);setInterval(() => {},1000)`
    const child = Bun.spawn([process.execPath, "-e", code], { detached: true, stdin: "ignore", stdout: "pipe", stderr: "inherit" })
    let enginePid: number | undefined
    try {
      const reader = child.stdout.getReader()
      const next = await reader.read()
      reader.releaseLock()
      enginePid = Number(new TextDecoder().decode(next.value).trim())
      expect(Number.isSafeInteger(enginePid) && enginePid > 0).toBe(true)
      const identity = Option.getOrThrow(await run(ProcessGroupControllerLive.inspect(child.pid)))
      const tree = await run(captureLegacyTree(LegacyOwner.make({ ...identity, port: 1234 })).pipe(
        Effect.provide([
          Layer.succeed(ProcessGroupController, ProcessGroupControllerLive),
          UnixLegacyProcessTable.pipe(Layer.provide(BunContext.layer)),
        ])))
      expect(tree.groups.map(group => group.leader.pid).sort((a, b) => a - b)).toEqual([child.pid, enginePid].sort((a, b) => a - b))
      expect(child.exitCode).toBeNull()
      // Only disposable fixtures are retired. Production migration wiring is separate.
      for (const group of tree.groups) expect((await run(ProcessGroupControllerLive.stop(group)))._tag).toBe("ProcessGroupStopped")
      for (const group of tree.groups) expect(await run(ProcessGroupControllerLive.waitForGroupExit(group, "2 seconds"))).toBe(true)
    } finally {
      for (const pid of [enginePid, child.pid]) if (pid !== undefined) {
        try { process.kill(-pid, "SIGKILL") } catch (error) { if (!(error instanceof Error && "code" in error && error.code === "ESRCH")) throw error }
      }
      await child.exited
    }
  }, 15000)
})
