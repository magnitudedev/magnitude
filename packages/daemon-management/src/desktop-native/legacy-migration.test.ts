import { mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { fileURLToPath } from "node:url"
import { Effect, Layer, Option, Schema } from "effect"
import { afterEach, beforeEach, describe, expect, it } from "vitest"
import { ProcessStartIdentitySchema } from "@magnitudedev/utils/process-groups"
import { LegacyMacStartup } from "./legacy-startup-macos"
import { LegacyLinuxStartup } from "./legacy-startup-linux"
import { fileLegacyMigrationJournal } from "./legacy-migration-journal"
import { LegacyMigrationActions, LegacyMigrationFailed, LegacyMigrationJournal, LegacyMigrationPlan, LegacyMigrationState, makeLegacyMigration } from "./legacy-migration"

let directory: string
beforeEach(async () => { directory = await mkdtemp(join(tmpdir(), "magnitude-migration-journal-")) })
afterEach(() => rm(directory, { recursive: true, force: true }))
const run = Effect.runPromise
const plan = LegacyMigrationPlan.make({
  startup: Option.some(LegacyMacStartup.make({ label: "dev.magnitude.acn", path: "/fixture/agent.plist", digest: LegacyMacStartup.fields.digest.make("a".repeat(64)), enabled: false, runningPid: Option.none() })),
  service: { _tag: "Captured", tree: {
    owner: { pid: 1234, processStartIdentity: ProcessStartIdentitySchema.make("legacy-owner"), port: 54321 },
    groups: [{ leader: { pid: 1234, processStartIdentity: ProcessStartIdentitySchema.make("legacy-owner") } }, { leader: { pid: 1235, processStartIdentity: ProcessStartIdentitySchema.make("legacy-engine") } }],
  } },
})
const readCheckpoint = async () => Schema.decodeUnknownSync(Schema.parseJson(LegacyMigrationState))(await readFile(join(directory, "legacy-migration.json"), "utf8"))
const fixture = (failAt?: string, source = plan) => {
  const calls: string[] = []
  let fail = true
  const step = (name: string) => Effect.gen(function* () {
    calls.push(name)
    if (fail && failAt === name) { fail = false; return yield* new LegacyMigrationFailed({ message: `interrupted ${name}` }) }
  })
  const actions = LegacyMigrationActions.of({
    prepare: step("prepare").pipe(Effect.as(source)),
    unregister: plan => { expect(Option.getOrThrow(plan.startup).enabled).toBe(Option.getOrThrow(source.startup).enabled); return step("unregister") },
    retire: service => { expect(service).toEqual(source.service); return step("retire") },
    transferLogin: enabled => { expect(enabled).toBe(Option.getOrThrow(source.startup).enabled); return step("login") },
    cleanup: service => { expect(service).toEqual(source.service); return step("cleanup") },
  })
  const create = () => run(makeLegacyMigration.pipe(Effect.provideService(LegacyMigrationActions, actions), Effect.provide(fileLegacyMigrationJournal(directory))))
  return { calls, create, actions }
}

describe("durable legacy migration", () => {
  it.each([false, true])("retains Linux login enabled=%s through a failed stage and fresh workflow", async enabled => {
    const source = { ...plan, startup: Option.some(LegacyLinuxStartup.make({
      unit: "magnitude.service", path: "/fixture/.config/systemd/user/magnitude.service",
      digest: LegacyLinuxStartup.fields.digest.make("b".repeat(64)), enabled, runningPid: Option.none(),
    })) }
    const f = fixture("unregister", source)
    await expect(run((await f.create()).run)).rejects.toThrow("interrupted unregister")
    const saved = await readCheckpoint()
    expect(saved._tag).toBe("Prepared")
    if (saved._tag === "Prepared") expect(saved.plan).toEqual(source)
    await run((await f.create()).run)
    expect((await readCheckpoint())._tag).toBe("Complete")
    expect(f.calls.filter(call => call === "prepare")).toHaveLength(1)
  })
  it("persists original intent before external mutation and finishes only after cleanup", async () => {
    const f = fixture()
    const original = f.actions.unregister
    const checked = { ...f.actions, unregister: (startup: LegacyMigrationPlan) => Effect.promise(async () => {
      const state = await readCheckpoint()
      expect(state._tag).toBe("Prepared")
      if (state._tag === "Prepared") expect(state.plan).toEqual(plan)
    }).pipe(Effect.zipRight(original(startup))) }
    const migration = await run(makeLegacyMigration.pipe(Effect.provideService(LegacyMigrationActions, checked), Effect.provide(fileLegacyMigrationJournal(directory))))
    await run(migration.run)
    expect(f.calls).toEqual(["prepare", "unregister", "retire", "login", "cleanup"])
    expect((await readCheckpoint())._tag).toBe("Complete")
    if (process.platform !== "win32") expect((await stat(join(directory, "legacy-migration.json"))).mode & 0o777).toBe(0o600)
    await run((await f.create()).run)
    expect(f.calls).toHaveLength(5)
  })
  it.each([
    ["unregister", "Prepared"], ["retire", "Unregistered"], ["login", "Retired"], ["cleanup", "LoginTransferred"],
  ])("resumes after %s failure without recomputing lost intent", async (step, checkpoint) => {
    const f = fixture(step)
    await expect(run((await f.create()).run)).rejects.toThrow(`interrupted ${step}`)
    expect((await readCheckpoint())._tag).toBe(checkpoint)
    await run((await f.create()).run)
    expect((await readCheckpoint())._tag).toBe("Complete")
    expect(f.calls.filter(call => call === "prepare")).toHaveLength(1)
    expect(f.calls.filter(call => call === step)).toHaveLength(2)
  })
  it("replays an idempotent action if its checkpoint failed after the action committed", async () => {
    const f = fixture()
    const journal = await run(Effect.map(LegacyMigrationJournal, value => value).pipe(Effect.provide(fileLegacyMigrationJournal(directory))))
    let fail = true
    const interrupted = Layer.succeed(LegacyMigrationJournal, { ...journal, write: (state: LegacyMigrationState) => {
      if (fail && state._tag === "Unregistered") { fail = false; return Effect.fail(new LegacyMigrationFailed({ message: "checkpoint write failed" })) }
      return journal.write(state)
    } })
    const migration = await run(makeLegacyMigration.pipe(Effect.provideService(LegacyMigrationActions, f.actions), Effect.provide(interrupted)))
    await expect(run(migration.run)).rejects.toThrow("checkpoint write failed")
    expect((await readCheckpoint())._tag).toBe("Prepared")
    await run((await f.create()).run)
    expect(f.calls.filter(call => call === "unregister")).toHaveLength(2)
    expect(f.calls.filter(call => call === "prepare")).toHaveLength(1)
  })
  it("does not overwrite current desktop login preference when no legacy registration existed", async () => {
    const f = fixture(undefined, { ...plan, startup: Option.none() })
    await run((await f.create()).run)
    expect(f.calls).toEqual(["prepare", "retire", "cleanup"])
  })
  it("serializes concurrent requests on the owning migration instance", async () => {
    const f = fixture()
    const migration = await f.create()
    await run(Effect.all([migration.run, migration.run], { concurrency: "unbounded" }))
    expect(f.calls).toEqual(["prepare", "unregister", "retire", "login", "cleanup"])
  })
  it("never resets a corrupt checkpoint or starts external actions", async () => {
    await writeFile(join(directory, "legacy-migration.json"), "corrupt")
    const f = fixture()
    await expect(run((await f.create()).run)).rejects.toThrow("checkpoint is malformed")
    expect(f.calls).toEqual([])
    expect(await readFile(join(directory, "legacy-migration.json"), "utf8")).toBe("corrupt")
  })
  it("resumes the original plan after the migration process is killed", async () => {
    const modulePath = fileURLToPath(new URL("./legacy-migration.ts", import.meta.url))
    const journalPath = fileURLToPath(new URL("./legacy-migration-journal.ts", import.meta.url))
    const source = `
      import {Effect,Schema} from 'effect';
      import {LegacyMigrationActions,LegacyMigrationPlan,makeLegacyMigration} from ${JSON.stringify(modulePath)};
      import {fileLegacyMigrationJournal} from ${JSON.stringify(journalPath)};
      const plan=Schema.decodeUnknownSync(LegacyMigrationPlan)(${JSON.stringify(Schema.encodeSync(LegacyMigrationPlan)(plan))});
      const forbidden=()=>Effect.dieMessage('later actions must not run');
      const actions=LegacyMigrationActions.of({prepare:Effect.succeed(plan),unregister:()=>Effect.gen(function*(){process.stdout.write('UNREGISTERED\\n');yield*Effect.never}),retire:forbidden,transferLogin:forbidden,cleanup:forbidden});
      await Effect.runPromise(Effect.flatMap(makeLegacyMigration,value=>value.run).pipe(Effect.provideService(LegacyMigrationActions,actions),Effect.provide(fileLegacyMigrationJournal(${JSON.stringify(directory)}))));
    `
    const child = Bun.spawn([process.execPath, "-e", source], { stdout: "pipe", stderr: "inherit", stdin: "ignore" })
    try {
      const reader = child.stdout.getReader()
      const first = await reader.read()
      reader.releaseLock()
      expect(new TextDecoder().decode(first.value)).toBe("UNREGISTERED\n")
      child.kill("SIGKILL")
      await child.exited
      expect((await readCheckpoint())._tag).toBe("Prepared")
      const f = fixture()
      await run((await f.create()).run)
      expect(f.calls).toEqual(["unregister", "retire", "login", "cleanup"])
      expect((await readCheckpoint())._tag).toBe("Complete")
    } finally {
      if (child.exitCode === null) child.kill("SIGKILL")
      await child.exited
    }
  }, 10000)
})
