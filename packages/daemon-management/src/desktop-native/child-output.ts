import type { Writable } from "node:stream"
import { Effect, Ref, Schema } from "effect"

export const ChildOutputMode = Schema.Literal("DiagnosticTail", "Foreground")
export type ChildOutputMode = typeof ChildOutputMode.Type

/** Diagnostic collection never waits for the terminal and never owns its lifetime. */
export const makeChildOutput = (mode: ChildOutputMode, sink: Writable = process.stderr) => Effect.gen(function* () {
  const tail = yield* Ref.make(Buffer.alloc(0))
  let enabled = mode === "Foreground"
  let pending = false
  let closed = false
  const failed = () => { enabled = false }
  if (enabled) yield* Effect.acquireRelease(
    Effect.sync(() => { sink.on("error", failed) }),
    () => Effect.sync(() => {
      closed = true
      enabled = false
      // A pending write may still emit error after child scope retirement.
      if (!pending) sink.removeListener("error", failed)
    }),
  )
  return {
    diagnosticTail: Ref.get(tail).pipe(Effect.map(bytes => bytes.toString("utf8"))),
    append: (chunk: Uint8Array) => Ref.update(tail, bytes => chunk.length >= 16_384
      ? Buffer.from(chunk.subarray(chunk.length - 16_384))
      : Buffer.concat([bytes.subarray(Math.max(0, bytes.length + chunk.length - 16_384)), chunk])).pipe(
      Effect.zipRight(Effect.sync(() => {
        if (!enabled || pending || sink.destroyed || sink.writableNeedDrain) return
        pending = true
        try {
          // At most one bounded write is outstanding; excess live output remains diagnostic-only.
          sink.write(Buffer.from(chunk.subarray(Math.max(0, chunk.length - 16_384))), error => {
            pending = false
            if (error) enabled = false
            if (closed) setImmediate(() => sink.removeListener("error", failed))
          })
        } catch {
          pending = false
          enabled = false
        }
      })),
    ),
  }
})
