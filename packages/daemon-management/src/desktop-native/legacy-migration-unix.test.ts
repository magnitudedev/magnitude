import { Database } from "bun:sqlite"
import { BunContext } from "@effect/platform-bun"
import { FetchHttpClient } from "@effect/platform"
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { Effect, Layer, Option } from "effect"
import { describe, expect, it } from "vitest"
import { ProcessGroupController } from "@magnitudedev/utils/process-groups"
import { ProcessGroupControllerLive } from "@magnitudedev/utils/process-groups/native"
import { BunSqliteDriverLayer } from "../bun"
import { LegacyStartupCommands } from "./legacy-startup-command"
import { macLegacyStartupLayer, linuxLegacyStartupLayer } from "./legacy-startup"
import { UnixLegacyProcessTable } from "./legacy-tree"
import { makeUnixLegacyMigrationActions } from "./legacy-migration-unix"
import { LegacyMigrationActions, makeLegacyMigration } from "./legacy-migration"
import { fileLegacyMigrationJournal } from "./legacy-migration-journal"

const run = Effect.runPromise
const withLegacy = async (use: (f: {
  root: string; database: string; pid: number; enginePid: number;
  actions: LegacyMigrationActions;
}) => Promise<void>, platform: "mac" | "linux" = "mac") => {
  const root = await mkdtemp(join(tmpdir(), "magnitude-retirement-"))
  const database = join(root, "acn/coordination.sqlite")
  await mkdir(join(root, "acn"))
  const engineCode = "Bun.spawn([process.execPath,'-e','setInterval(()=>{},1000)'],{stdin:'ignore',stdout:'ignore',stderr:'ignore'});setInterval(()=>{},1000)"
  const code = `const engine=Bun.spawn([process.execPath,'-e',${JSON.stringify(engineCode)}],{detached:true,stdin:'ignore',stdout:'ignore',stderr:'ignore'});const server=Bun.serve({hostname:'127.0.0.1',port:0,fetch:()=>Response.json({service:'magnitude-acn',pid:process.pid})});console.log(JSON.stringify({port:server.port,enginePid:engine.pid}));process.on('SIGTERM',()=>process.exit(0));`
  const child = Bun.spawn([process.execPath, "-e", code], { detached: true, stdin: "ignore", stdout: "pipe", stderr: "inherit" })
  let enginePid: number | undefined
  try {
    const reader = child.stdout.getReader()
    const first = await reader.read()
    reader.releaseLock()
    const info = JSON.parse(new TextDecoder().decode(first.value)) as { port: number; enginePid: number }
    enginePid = info.enginePid
    const identity = Option.getOrThrow(await run(ProcessGroupControllerLive.inspect(child.pid)))
    const db = new Database(database, { create: true })
    db.exec("CREATE TABLE owner (id INTEGER, pid INTEGER, process_start_identity TEXT, port INTEGER)")
    db.query("INSERT INTO owner VALUES (1, ?, ?, ?)").run(identity.pid, identity.processStartIdentity, info.port)
    db.close()
    const commands = LegacyStartupCommands.of({ run: (_, args) => args[0] === "print"
      ? Effect.succeed({ code: 113, stdout: "", stderr: "not found" }) : Effect.dieMessage("Fixture has no startup registration to mutate") })
    const actions = await run(makeUnixLegacyMigrationActions({ dataDirectory: root,
      transferLogin: () => Effect.dieMessage("No legacy registration means no login change"),
    }).pipe(Effect.provide([
      (platform === "mac" ? macLegacyStartupLayer(root) : linuxLegacyStartupLayer({ home: root, runtimeDirectory: root }))
        .pipe(Layer.provide(Layer.succeed(LegacyStartupCommands, commands))),
      Layer.succeed(ProcessGroupController, ProcessGroupControllerLive),
      UnixLegacyProcessTable.pipe(Layer.provide(BunContext.layer)),
      BunSqliteDriverLayer, FetchHttpClient.layer, BunContext.layer,
    ])))
    await use({ root, database, pid: child.pid, enginePid, actions })
  } finally {
    for (const pid of [enginePid, child.pid]) if (pid !== undefined) {
      try { process.kill(-pid, "SIGKILL") } catch (error) { if (!(error instanceof Error && "code" in error && error.code === "ESRCH")) throw error }
    }
    await child.exited
    await rm(root, { recursive: true, force: true })
  }
}

describe.skipIf(process.platform === "win32")("native legacy migration actions", () => {
  it.each(["mac", "linux"] as const)("%s composition retires separate groups and preserves application data", async platform => withLegacy(async f => {
    await mkdir(join(f.root, "models"))
    await writeFile(join(f.root, "models/keep"), "model data")
    await writeFile(join(f.root, "settings.json"), "settings")
    const plan = await run(f.actions.prepare)
    expect(plan.service._tag).toBe("Captured")
    if (plan.service._tag !== "Captured") throw new Error("Expected a live legacy fixture")
    expect(plan.service.tree.groups.map(group => group.leader.pid).sort((a, b) => a - b)).toEqual([f.pid, f.enginePid].sort((a, b) => a - b))
    const migration = await run(makeLegacyMigration.pipe(Effect.provideService(LegacyMigrationActions, f.actions), Effect.provide(fileLegacyMigrationJournal(join(f.root, "desktop")))))
    await run(migration.run)
    for (const group of plan.service.tree.groups) expect((await run(ProcessGroupControllerLive.observe(group)))._tag).toBe("ProcessGroupAbsent")
    await expect(readFile(f.database)).rejects.toMatchObject({ code: "ENOENT" })
    expect(await readFile(join(f.root, "models/keep"), "utf8")).toBe("model data")
    expect(await readFile(join(f.root, "settings.json"), "utf8")).toBe("settings")
    await run(migration.run)
  }, platform), 15000)
  it("does not signal or delete when the saved owner record changes", async () => withLegacy(async f => {
    const plan = await run(f.actions.prepare)
    const db = new Database(f.database)
    db.exec("UPDATE owner SET process_start_identity = 'different-occurrence'")
    db.close()
    await expect(run(f.actions.retire(plan.service))).rejects.toThrow("ownership changed")
    await expect(run(f.actions.cleanup(plan.service))).rejects.toThrow("ownership changed")
    expect(Option.isSome(await run(ProcessGroupControllerLive.inspect(f.pid)))).toBe(true)
    expect(Option.isSome(await run(ProcessGroupControllerLive.inspect(f.enginePid)))).toBe(true)
    expect((await readFile(f.database)).length).toBeGreaterThan(0)
  }), 15000)
  it("refuses database cleanup until every captured group is absent", async () => withLegacy(async f => {
    const plan = await run(f.actions.prepare)
    await expect(run(f.actions.cleanup(plan.service))).rejects.toThrow("absence is unproven")
    expect(Option.isSome(await run(ProcessGroupControllerLive.inspect(f.pid)))).toBe(true)
  }), 15000)
})
