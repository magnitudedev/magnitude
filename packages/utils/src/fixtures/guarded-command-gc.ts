import { Effect } from "effect"
import { closeSync, fstatSync, openSync } from "node:fs"
import { GuardedCommand, guardedCommandLayer } from "../guarded-command"

// Run outside the test worker: a stale pipe descriptor may already belong to SQLite,
// in which case macOS kills the process before JavaScript can report the violation.
await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const command = yield* GuardedCommand
  for (let index = 0; index < 3; index++) {
    yield* command.run("/bin/sh", ["-c", "printf output"], {})
  }
  const descriptors = yield* Effect.acquireRelease(
    Effect.sync(() => Array.from({ length: 32 }, () => openSync("/dev/null", "r"))),
    descriptors => Effect.sync(() => {
      for (const descriptor of descriptors) { try { closeSync(descriptor) } catch {} }
    }),
  )
  // Reuse the helpers' retired pipe numbers before their Subprocess objects are collected.
  for (let index = 0; index < 5; index++) {
    yield* Effect.sync(() => Bun.gc(true))
    yield* Effect.sleep("10 millis")
  }
  yield* Effect.sync(() => {
    for (const descriptor of descriptors) fstatSync(descriptor)
  })
})).pipe(Effect.provide(guardedCommandLayer(process.argv[2]!))))
