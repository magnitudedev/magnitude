import * as Command from "@effect/platform/Command"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import { Context, Effect, Layer, Option, Schema } from "effect"
import { ProcessGroupController, ProcessGroupSchema } from "@magnitudedev/utils/process-groups"
import { LegacyOwner } from "./legacy-owner"

const Pid = Schema.Int.pipe(Schema.between(1, Number.MAX_SAFE_INTEGER))
const ProcessRow = Schema.Struct({ pid: Pid, parent: Schema.Int.pipe(Schema.nonNegative()), group: Pid })
export type LegacyProcessRow = typeof ProcessRow.Type
export class LegacyTreeUnproven extends Schema.TaggedError<LegacyTreeUnproven>()("LegacyTreeUnproven", { message: Schema.String }) {}
export class LegacyTreeChanged extends Schema.TaggedError<LegacyTreeChanged>()("LegacyTreeChanged", {}) {
  override get message() { return "Legacy process groups changed during inspection" }
}

export interface LegacyProcessTable {
  readonly read: Effect.Effect<readonly LegacyProcessRow[], LegacyTreeUnproven>
}
export const LegacyProcessTable = Context.GenericTag<LegacyProcessTable>("@magnitudedev/daemon-management/LegacyProcessTable")

export const UnixLegacyProcessTable = Layer.effect(LegacyProcessTable, Effect.gen(function* () {
  const executor = yield* CommandExecutor.CommandExecutor
  return LegacyProcessTable.of({
  read: Effect.gen(function* () {
    if (process.platform !== "darwin" && process.platform !== "linux") return yield* new LegacyTreeUnproven({ message: "Legacy process-tree inspection requires a native platform adapter" })
    const output = yield* Command.make("/bin/ps", "-axo", "pid=,ppid=,pgid=").pipe(Command.string,
      Effect.mapError(error => new LegacyTreeUnproven({ message: `Cannot inspect legacy process tree: ${String(error)}` })))
    const rows = output.trim().split("\n").filter(line => line.trim()).map(line => {
      const [pid, parent, group] = line.trim().split(/\s+/).map(Number)
      return { pid, parent, group }
    })
    return yield* Schema.decodeUnknown(Schema.Array(ProcessRow))(rows).pipe(
      Effect.mapError(() => new LegacyTreeUnproven({ message: "Native process table is malformed" })))
  }).pipe(Effect.provideService(CommandExecutor.CommandExecutor, executor)),
  })
}))

export const LegacyTree = Schema.Struct({ owner: LegacyOwner, groups: Schema.NonEmptyArray(ProcessGroupSchema) })
export type LegacyTree = typeof LegacyTree.Type

/** Selection is ancestry-based. A descendant in an unrelated group cannot authorize group signalling. */
export const selectLegacyGroups = (owner: LegacyOwner, rows: readonly LegacyProcessRow[]) => Effect.gen(function* () {
  const byPid = new Map(rows.map(row => [row.pid, row]))
  if (byPid.size !== rows.length) return yield* new LegacyTreeUnproven({ message: "Native process table contains duplicate identities" })
  const root = byPid.get(owner.pid)
  if (!root || root.group !== owner.pid) return yield* new LegacyTreeUnproven({ message: "Legacy owner is absent or does not lead its process group" })
  const selected = new Set([owner.pid])
  for (;;) {
    const before = selected.size
    for (const row of rows) if (selected.has(row.parent)) selected.add(row.pid)
    if (selected.size === before) break
  }
  const groups = new Set<number>()
  for (const pid of selected) {
    const group = byPid.get(pid)!.group
    if (!selected.has(group) || byPid.get(group)?.group !== group) {
      return yield* new LegacyTreeUnproven({ message: "Legacy descendant belongs to an unverified process group" })
    }
    groups.add(group)
  }
  // An unrelated member sharing an owned group is not ours to signal.
  if (rows.some(row => groups.has(row.group) && !selected.has(row.pid))) {
    return yield* new LegacyTreeUnproven({ message: "Legacy process group contains an unrelated process" })
  }
  return [...groups].sort((a, b) => a - b)
})

/** Capture only while the exact live owner and group topology remain stable. No signalling here. */
export const captureLegacyTree = (owner: LegacyOwner) => Effect.gen(function* () {
  const processes = yield* ProcessGroupController
  const table = yield* LegacyProcessTable
  const verifyOwner = processes.inspect(owner.pid).pipe(Effect.flatMap(current =>
    Option.exists(current, value => value.processStartIdentity === owner.processStartIdentity)
      ? Effect.void : Effect.fail(new LegacyTreeUnproven({ message: "Legacy owner identity changed or exited" }))))
  yield* verifyOwner
  const pids = yield* selectLegacyGroups(owner, yield* table.read)
  const groups = yield* Effect.forEach(pids, pid => processes.inspect(pid).pipe(Effect.flatMap(identity =>
    Option.isSome(identity) ? Effect.succeed({ leader: identity.value })
      : Effect.fail(new LegacyTreeUnproven({ message: "Legacy group leader exited during inspection" })))))
  yield* verifyOwner
  const after = yield* selectLegacyGroups(owner, yield* table.read)
  if (pids.length !== after.length || pids.some((pid, index) => pid !== after[index])) {
    return yield* new LegacyTreeChanged()
  }
  for (const group of groups) {
    const current = yield* processes.inspect(group.leader.pid)
    if (!Option.exists(current, value => value.processStartIdentity === group.leader.processStartIdentity)) {
      return yield* new LegacyTreeUnproven({ message: "Legacy group identity changed during inspection" })
    }
  }
  return yield* Schema.decodeUnknown(LegacyTree)({ owner, groups }).pipe(
    Effect.mapError(() => new LegacyTreeUnproven({ message: "Legacy process tree is empty" })))
})
