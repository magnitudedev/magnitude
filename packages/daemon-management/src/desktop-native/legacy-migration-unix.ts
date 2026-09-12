import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import { unlink } from "node:fs/promises"
import { join } from "node:path"
import { Effect, Option, Schema, Schedule } from "effect"
import { ProcessGroupController } from "@magnitudedev/utils/process-groups"
import { SqliteDriver } from "../sqlite-driver"
import { LegacyOwner, readLegacyOwner } from "./legacy-owner"
import { captureLegacyTree, LegacyProcessTable } from "./legacy-tree"
import { LegacyStartup } from "./legacy-startup"
import { LegacyMigrationActions, LegacyMigrationFailed, type LegacyServiceSource } from "./legacy-migration"

const sameOwner = Schema.equivalence(LegacyOwner)
const failed = (error: { readonly message: string }) => new LegacyMigrationFailed({ message: error.message })
const privateHealth = Schema.Struct({ service: Schema.Literal("magnitude-acn"), pid: LegacyOwner.fields.pid })
const missing = (error: unknown) => error instanceof Error && "code" in error && error.code === "ENOENT"

/** Composition uses the new migration primitives, never the old manager or owner-election store. */
export const makeUnixLegacyMigrationActions = (options: {
  readonly dataDirectory: string
  readonly transferLogin: (enabled: boolean) => Effect.Effect<void, LegacyMigrationFailed>
}) => Effect.gen(function* () {
  const processes = yield* ProcessGroupController
  const table = yield* LegacyProcessTable
  const sqlite = yield* SqliteDriver
  const http = yield* HttpClient.HttpClient
  const startup = yield* LegacyStartup
  const read = readLegacyOwner(options.dataDirectory).pipe(Effect.provideService(SqliteDriver, sqlite), Effect.mapError(failed))
  const capture = (owner: LegacyOwner) => captureLegacyTree(owner).pipe(
    Effect.provideService(ProcessGroupController, processes), Effect.provideService(LegacyProcessTable, table),
    Effect.retry({ while: error => error._tag === "LegacyTreeChanged", times: 3, schedule: Schedule.spaced("50 millis") }),
    Effect.mapError(failed))
  const recorded = (source: LegacyServiceSource) => source._tag === "Missing" ? Option.none<LegacyOwner>()
    : Option.some(source._tag === "Captured" ? source.tree.owner : source.owner)
  const confirmRecord = (source: LegacyServiceSource, allowRemoved = false) => Effect.gen(function* () {
    const current = yield* read
    if (allowRemoved && Option.isNone(current)) return
    const expected = recorded(source)
    if (Option.isNone(expected) ? Option.isSome(current) : !Option.exists(current, value => sameOwner(value, expected.value))) {
      return yield* new LegacyMigrationFailed({ message: "Legacy ownership changed after migration was prepared; no new service will start" })
    }
  })
  const confirmTree = (source: LegacyServiceSource) => Effect.gen(function* () {
    if (source._tag !== "Captured") return
    const ownerGroup = source.tree.groups.find(group => group.leader.pid === source.tree.owner.pid && group.leader.processStartIdentity === source.tree.owner.processStartIdentity)
    if (!ownerGroup || new Set(source.tree.groups.map(group => group.leader.pid)).size !== source.tree.groups.length) {
      return yield* new LegacyMigrationFailed({ message: "Saved legacy tree does not contain one verified owner group" })
    }
    const root = yield* processes.observe(ownerGroup).pipe(Effect.mapError(failed))
    if (root._tag === "ProcessGroupLeaderReplaced") return yield* new LegacyMigrationFailed({ message: "Legacy owner PID now belongs to another process" })
    if (root._tag === "ProcessGroupLeaderLive") {
      const current = yield* capture(source.tree.owner)
      if (current.groups.length !== source.tree.groups.length || current.groups.some(group => !source.tree.groups.some(saved => saved.leader.pid === group.leader.pid && saved.leader.processStartIdentity === group.leader.processStartIdentity))) {
        return yield* new LegacyMigrationFailed({ message: "Legacy process tree changed after migration was prepared" })
      }
    }
  })
  const requireAbsent = (source: LegacyServiceSource) => Effect.gen(function* () {
    const groups = source._tag === "Captured" ? source.tree.groups : source._tag === "Absent" ? [{ leader: source.owner }] : []
    for (const group of groups) if ((yield* processes.observe(group).pipe(Effect.mapError(failed)))._tag !== "ProcessGroupAbsent") {
      return yield* new LegacyMigrationFailed({ message: "Legacy process-group absence is unproven" })
    }
  })
  return LegacyMigrationActions.of({
    prepare: Effect.gen(function* () {
      const registration = yield* startup.inspect.pipe(Effect.mapError(failed))
      const owner = yield* read
      const registeredPid = Option.flatMap(registration, value => value.runningPid)
      if (Option.isSome(registeredPid) && !Option.exists(owner, value => value.pid === registeredPid.value)) {
        return yield* new LegacyMigrationFailed({ message: "Legacy startup registration is running without a matching owner record" })
      }
      if (Option.isNone(owner)) return { startup: registration, service: { _tag: "Missing" as const } }
      const observed = yield* processes.observe({ leader: owner.value }).pipe(Effect.mapError(failed))
      if (observed._tag === "ProcessGroupAbsent") {
        if (Option.isSome(registeredPid)) return yield* new LegacyMigrationFailed({ message: "Legacy startup process identity is unproven" })
        return { startup: registration, service: { _tag: "Absent" as const, owner: owner.value } }
      }
      if (observed._tag !== "ProcessGroupLeaderLive") return yield* new LegacyMigrationFailed({ message: "Legacy owner identity or descendant ancestry is unproven" })
      const health = yield* http.execute(HttpClientRequest.get(`http://127.0.0.1:${owner.value.port}/health`)).pipe(
        Effect.flatMap(response => response.json), Effect.flatMap(Schema.decodeUnknown(privateHealth)),
        Effect.timeoutFail({ duration: "2 seconds", onTimeout: () => new LegacyMigrationFailed({ message: "Legacy service identity check timed out" }) }),
        Effect.mapError(error => new LegacyMigrationFailed({ message: `Could not verify legacy service identity: ${String(error)}` })))
      if (health.pid !== owner.value.pid) return yield* new LegacyMigrationFailed({ message: "Legacy health belongs to another process" })
      const tree = yield* capture(owner.value)
      const service = { _tag: "Captured" as const, tree }
      yield* confirmRecord(service)
      return { startup: registration, service }
    }),
    unregister: plan => confirmRecord(plan.service).pipe(Effect.zipRight(confirmTree(plan.service)), Effect.zipRight(
      Option.isSome(plan.startup) ? startup.unregister(plan.startup.value).pipe(Effect.mapError(failed)) : Effect.void)),
    retire: source => Effect.gen(function* () {
      yield* confirmRecord(source)
      yield* confirmTree(source)
      if (source._tag === "Captured") {
        // Legacy ACN handles SIGTERM as graceful shutdown. Stop the root before residual ICN groups.
        const groups = [...source.tree.groups].sort((a, b) => Number(b.leader.pid === source.tree.owner.pid) - Number(a.leader.pid === source.tree.owner.pid))
        for (const group of groups) {
          yield* confirmRecord(source)
          const result = yield* processes.stop(group).pipe(Effect.mapError(failed))
          if (result._tag !== "ProcessGroupStopped") return yield* new LegacyMigrationFailed({ message: "A saved legacy PID was reused; it has not been signalled" })
        }
      }
      yield* requireAbsent(source)
    }),
    transferLogin: options.transferLogin,
    cleanup: source => Effect.gen(function* () {
      yield* confirmRecord(source, true)
      yield* requireAbsent(source)
      const database = join(options.dataDirectory, "acn/coordination.sqlite")
      for (const path of [database, `${database}-journal`, `${database}-wal`, `${database}-shm`]) {
        yield* Effect.tryPromise({ try: () => unlink(path).catch(error => { if (!missing(error)) throw error }),
          catch: error => new LegacyMigrationFailed({ message: `Could not remove retired legacy database: ${String(error)}` }) })
      }
    }),
  })
})
