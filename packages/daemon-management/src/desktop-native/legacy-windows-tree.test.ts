import { Effect, Option, Schema, type Scope } from "effect"
import { describe, expect, it } from "vitest"
import { captureLegacyWindowsDescendants } from "./legacy-windows-tree"
import { WindowsObservedProcess, WindowsProcessObserver, WindowsProcessParent } from "./windows-process-observer"
const identity = (pid: number, created = pid, user = "S-1-5-21-1-2-3-1001") => Schema.decodeUnknownSync(WindowsObservedProcess)({ pid, creationTime: created.toString(16).padStart(16, "0"), executable: `C:\\Magnitude\\${pid}.exe`, userSid: user })
const rows = (...entries: [number, number][]) => Schema.decodeUnknownSync(Schema.Array(WindowsProcessParent))(entries.map(([pid, parentPid]) => ({ pid, parentPid })))
const initial = rows([10, 1], [20, 10], [30, 20], [99, 1])
const fixture = () => {
  let table = initial
  const identities = new Map(initial.map(row => [row.pid, identity(row.pid)]))
  const exited = new Set<number>(), missing = new Set<number>(), acquired: number[] = [], released: number[] = []
  const observer = WindowsProcessObserver.of({
    snapshotParents: Effect.sync(() => table),
    observe: pid => missing.has(pid) ? Effect.succeed(Option.none()) : Effect.acquireRelease(
      Effect.sync(() => { acquired.push(pid); return Option.some({ details: Effect.sync(() => identities.get(pid)!), exited: Effect.sync(() => exited.has(pid)), awaitExit: Effect.void }) }),
      () => Effect.sync(() => { released.push(pid) }),
    ),
  })
  return { observer, identities, exited, missing, acquired, released, setRows: (next: typeof initial) => { table = next } }
}
const run = <A, E>(f: ReturnType<typeof fixture>, action: Effect.Effect<A, E, Scope.Scope | WindowsProcessObserver>) => Effect.runPromise(Effect.scoped(action.pipe(Effect.provideService(WindowsProcessObserver, f.observer))))
describe("read-only legacy Windows descendant capture", () => {
  it("retains exactly the root and descendants, excluding unrelated processes", async () => {
    const f = fixture()
    await run(f, Effect.gen(function* () {
      const capture = yield* captureLegacyWindowsDescendants(identity(10))
      expect(capture.snapshot.processes.map(row => row.identity.pid)).toEqual([10, 20, 30])
      expect(f.released).toEqual([])
      yield* capture.verify
    }))
    expect(f.acquired).toEqual([10, 20, 30])
    expect(f.released.sort()).toEqual([10, 20, 30])
  })
  it.each(["root", "user", "parent", "missing", "cycle", "absent root", "duplicate", "exited"] as const)("rejects unproven %s evidence", async mode => {
    const f = fixture()
    if (mode === "root") f.identities.set(identity(10).pid, identity(10, 11))
    if (mode === "user") f.identities.set(identity(20).pid, identity(20, 20, "S-1-5-21-1-2-3-1002"))
    if (mode === "parent") f.identities.set(identity(20).pid, identity(20, 9))
    if (mode === "missing") f.missing.add(20)
    if (mode === "cycle") f.setRows(rows([10, 30], [20, 10], [30, 20]))
    if (mode === "absent root") f.setRows(rows([20, 10]))
    if (mode === "duplicate") f.setRows(rows([10, 1], [20, 10], [20, 10]))
    if (mode === "exited") f.exited.add(20)
    expect((await run(f, Effect.either(captureLegacyWindowsDescendants(identity(10)))))._tag).toBe("Left")
    expect(f.released.sort()).toEqual(f.acquired.sort())
  })
  it.each(["new child", "changed parent", "exit"] as const)("invalidates evidence after %s", async mode => {
    const f = fixture()
    await run(f, Effect.gen(function* () {
      const captured = yield* captureLegacyWindowsDescendants(identity(10))
      if (mode === "new child") f.setRows(rows([10, 1], [20, 10], [30, 20], [40, 10]))
      if (mode === "changed parent") f.setRows(rows([10, 1], [20, 10], [30, 10]))
      if (mode === "exit") f.exited.add(20)
      expect((yield* Effect.either(captured.verify))._tag).toBe("Left")
    }))
    expect(f.released.sort()).toEqual(f.acquired.sort())
  })
})
