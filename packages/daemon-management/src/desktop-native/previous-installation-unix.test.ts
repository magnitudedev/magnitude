import { Database } from "bun:sqlite"
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { Effect, Option, Schema } from "effect"
import { afterEach, beforeEach, describe, expect, it } from "vitest"
import { ProcessGroupController, ProcessStartIdentitySchema } from "@magnitudedev/utils/process-groups"
import { ProcessGroupControllerLive } from "@magnitudedev/utils/process-groups/native"
import { BunSqliteDriverLayer } from "../bun"
import { LegacyStartup, LegacyStartupRegistration } from "./legacy-startup"
import { NativeLegacyStartupCommands, LegacyStartupCommands } from "./legacy-startup-command"
import { LegacyProcessTable } from "./legacy-tree"
import { makeUnixPreviousInstallation } from "./previous-installation-unix"

let home: string
let dataDirectory: string
beforeEach(async () => { home = await mkdtemp(join(tmpdir(), "magnitude-upgrade-inspection-")); dataDirectory = join(home, ".magnitude"); await mkdir(join(dataDirectory, "acn"), { recursive: true }) })
afterEach(() => rm(home, { recursive: true, force: true }))
const forbidden = Effect.dieMessage("Unverified processes must not be changed")
const make = (registration = Option.none<LegacyStartupRegistration>()) => Effect.runPromise(makeUnixPreviousInstallation({ home, dataDirectory }).pipe(
  Effect.provide([BunSqliteDriverLayer, NativeLegacyStartupCommands]),
  Effect.provideService(LegacyProcessTable, { read: forbidden }),
  Effect.provideService(ProcessGroupController, { ...ProcessGroupControllerLive, stop: () => forbidden }),
  Effect.provideService(LegacyStartup, { inspect: Effect.succeed(registration), unregister: () => Effect.void })))
const record = (pid: number, identity: string) => {
  const db = new Database(join(dataDirectory, "acn/coordination.sqlite"), { create: true })
  db.exec("CREATE TABLE owner (id INTEGER, pid INTEGER, process_start_identity TEXT, port INTEGER)")
  db.query("INSERT INTO owner VALUES (1, ?, ?, 54321)").run(pid, identity); db.close()
}

describe.skipIf(process.platform === "win32")("Unix upgrade admission", () => {

  it.skipIf(process.platform !== "darwin")("saves newly started inference groups before stopping and retains orphaned saved groups", async () => {
    const executable = join(dataDirectory, "releases/acn/old/magnitude-service")
    await mkdir(join(executable, ".."), { recursive: true }); await writeFile(executable, "fixture")
    const identity = (pid: number) => ({ pid, processStartIdentity: ProcessStartIdentitySchema.make(`start-${pid}`) })
    const events: string[] = []
    const controller = { ...ProcessGroupControllerLive,
      inspect: (pid: number) => Effect.succeed(Option.some(identity(pid))),
      stop: (group: Parameters<ProcessGroupController["stop"]>[0]) => Effect.sync(() => { events.push(`stop-${group.leader.pid}`); return { _tag: "ProcessGroupStopped" as const, group } }),
      observe: (group: Parameters<ProcessGroupController["observe"]>[0]) => Effect.succeed({ _tag: "ProcessGroupAbsent" as const, group }),
    }
    const installation = await Effect.runPromise(makeUnixPreviousInstallation({ home, dataDirectory }).pipe(
      Effect.provide(BunSqliteDriverLayer),
      Effect.provideService(ProcessGroupController, controller),
      Effect.provideService(LegacyStartupCommands, { run: () => Effect.succeed({ code: 0, stdout: `${process.getuid!()} ${executable}`, stderr: "" }) }),
      Effect.provideService(LegacyProcessTable, { read: Effect.succeed([{ pid: 100, parent: 1, group: 100 }, { pid: 102, parent: 100, group: 102 }]) }),
      Effect.provideService(LegacyStartup, { inspect: Effect.succeed(Option.none()), unregister: () => forbidden })))
    const plan = { _tag: "Unix" as const, startup: Option.none(), tree: Option.some({ owner: { ...identity(100), port: 54321 }, groups: [{ leader: identity(100) }, { leader: identity(101) }] as const }) }
    await Effect.runPromise(installation.retire(plan, updated => Effect.sync(() => {
      events.push("checkpoint")
      expect(Option.getOrThrow(updated.tree).groups.map(group => group.leader.pid)).toEqual([100, 101, 102])
    })))
    expect(events).toEqual(["checkpoint", "stop-100", "stop-101", "stop-102"])
  })
  it("leaves a clean home alone", async () => { expect(Option.isNone(await Effect.runPromise((await make()).inspect))).toBe(true) })
  it("ignores corrupt obsolete metadata without repairing it", async () => {
    const path = join(dataDirectory, "acn/coordination.sqlite"); await writeFile(path, "old broken metadata")
    expect(Option.isNone(await Effect.runPromise((await make()).inspect))).toBe(true)
    expect(await readFile(path, "utf8")).toBe("old broken metadata")
  })
  it("does not claim a reused PID", async () => {
    record(process.pid, "an earlier process")
    expect(Option.isNone(await Effect.runPromise((await make()).inspect))).toBe(true)
  })
  it("rejects an unrelated executable even with an exact PID and start identity", async () => {
    const identity = Option.getOrThrow(await Effect.runPromise(ProcessGroupControllerLive.inspect(process.pid)))
    record(identity.pid, identity.processStartIdentity)
    await expect(Effect.runPromise((await make()).inspect)).rejects.toThrow("does not match a Magnitude installation")
  })
  it("retires dormant registrations even without ownership metadata", async () => {
    const registration = Schema.decodeUnknownSync(LegacyStartupRegistration)({ _tag: "MacLaunchAgent", label: "dev.magnitude.acn", path: join(home, "Library/LaunchAgents/dev.magnitude.acn.plist"), digest: "a".repeat(64), enabled: false })
    const installation = await make(Option.some(registration))
    const plan = Option.getOrThrow(await Effect.runPromise(installation.inspect))
    expect(Option.isNone(plan.tree)).toBe(true)
    await Effect.runPromise(installation.retire(plan, () => forbidden))
  })
})
