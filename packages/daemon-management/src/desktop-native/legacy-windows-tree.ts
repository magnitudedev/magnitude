import { Effect, Option, Schema } from "effect"
import { WindowsProcessId } from "@magnitudedev/utils/windows-native"
import { WindowsObservedProcess, WindowsProcessObserver, WindowsProcessParent } from "./windows-process-observer"

export class LegacyWindowsTreeUnproven extends Schema.TaggedError<LegacyWindowsTreeUnproven>()("LegacyWindowsTreeUnproven", { message: Schema.String }) {}
export const LegacyWindowsDescendants = Schema.Struct({
  root: WindowsObservedProcess,
  processes: Schema.NonEmptyArray(Schema.Struct({ identity: WindowsObservedProcess, parentPid: WindowsProcessParent.fields.parentPid })),
})

const select = (root: WindowsProcessId, rows: readonly (typeof WindowsProcessParent.Type)[]) => Effect.gen(function* () {
  const byPid = new Map(rows.map(row => [row.pid, row]))
  if (byPid.size !== rows.length || !byPid.has(root)) return yield* new LegacyWindowsTreeUnproven({ message: "Legacy root is absent or process identities are ambiguous." })
  const children = new Map<number, (typeof WindowsProcessParent.Type)[]>()
  for (const row of rows) {
    const siblings = children.get(row.parentPid) ?? []
    siblings.push(row)
    children.set(row.parentPid, siblings)
  }
  const selected = [byPid.get(root)!]
  const seen = new Set<number>([root])
  for (let index = 0; index < selected.length; index++) {
    for (const child of children.get(selected[index]!.pid) ?? []) {
      if (seen.has(child.pid)) return yield* new LegacyWindowsTreeUnproven({ message: "Legacy process ancestry contains a cycle." })
      seen.add(child.pid)
      selected.push(child)
    }
  }
  return selected
})

/** Retains read-only handles in the caller's scope. This observes current descendants;
 * it cannot prove absence of historical orphans or authorize migration retirement. */
export const captureLegacyWindowsDescendants = (expected: WindowsObservedProcess) => Effect.gen(function* () {
  const observer = yield* WindowsProcessObserver
  const before = yield* select(expected.pid, yield* observer.snapshotParents)
  const retained = yield* Effect.forEach(before, row => Effect.gen(function* () {
    const observation = yield* observer.observe(row.pid)
    if (Option.isNone(observation)) return yield* new LegacyWindowsTreeUnproven({ message: "A legacy process exited during descendant capture." })
    const identity = yield* observation.value.details
    if (identity.userSid !== expected.userSid) return yield* new LegacyWindowsTreeUnproven({ message: "Legacy descendants include another user's process." })
    return { identity, parentPid: row.parentPid, observation: observation.value }
  }))
  const root = retained[0]!.identity
  if (root.creationTime !== expected.creationTime || root.executable !== expected.executable) {
    return yield* new LegacyWindowsTreeUnproven({ message: "Legacy root identity changed before descendant capture." })
  }
  const byPid = new Map(retained.map(row => [row.identity.pid, row]))
  for (const row of retained.slice(1)) {
    const parent = byPid.get(WindowsProcessId.make(row.parentPid))!
    if (BigInt(`0x${row.identity.creationTime}`) < BigInt(`0x${parent.identity.creationTime}`)) {
      return yield* new LegacyWindowsTreeUnproven({ message: "Legacy ancestry refers to a reused parent PID." })
    }
  }
  const verify = Effect.gen(function* () {
    const after = yield* select(expected.pid, yield* observer.snapshotParents)
    if (after.length !== retained.length || after.some(row => byPid.get(row.pid)?.parentPid !== row.parentPid)) {
      return yield* new LegacyWindowsTreeUnproven({ message: "Legacy descendants changed during observation." })
    }
    for (const row of retained) if (yield* row.observation.exited) {
      return yield* new LegacyWindowsTreeUnproven({ message: "A retained legacy process exited during observation." })
    }
  }).pipe(Effect.mapError(error => error instanceof LegacyWindowsTreeUnproven ? error : new LegacyWindowsTreeUnproven({ message: "Legacy process observation failed; ancestry is unproven." })))
  yield* verify
  const snapshot = yield* Schema.decodeUnknown(LegacyWindowsDescendants)({ root, processes: retained.map(({ identity, parentPid }) => ({ identity, parentPid })) })
  return { snapshot, verify }
}).pipe(Effect.mapError(error => error instanceof LegacyWindowsTreeUnproven ? error : new LegacyWindowsTreeUnproven({ message: "Could not capture legacy Windows descendants." })))
