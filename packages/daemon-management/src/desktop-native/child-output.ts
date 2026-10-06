import type { Writable } from "node:stream"
import { Effect, Option, Ref, Schema } from "effect"
import { openLogFile, type LogFile } from "@magnitudedev/utils/log-file"

export const ChildOutputMode = Schema.Literal("DiagnosticTail", "Foreground")
export type ChildOutputMode = typeof ChildOutputMode.Type

const LOG_FILE_BYTES = 10 * 1024 * 1024

/** Opens the persistent log a child's output is recorded in, when its command names one. */
export const openChildLog = (file: Option.Option<string>) => Option.match(file, {
  onNone: () => Effect.succeed(Option.none<LogFile>()),
  onSome: path => openLogFile(path, LOG_FILE_BYTES).pipe(Effect.map(Option.some)),
})

/** Diagnostic collection never waits for the terminal or the log file and never owns its lifetime. */
export const makeChildOutput = (mode: ChildOutputMode, log: Option.Option<LogFile>, sink: Writable = process.stderr) => Effect.gen(function* () {
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
  const logged = (text: string | Uint8Array) => Option.match(log, { onNone: () => Effect.void, onSome: file => file.append(text) })
  return {
    diagnosticTail: Ref.get(tail).pipe(Effect.map(bytes => bytes.toString("utf8"))),
    record: (note: string) => logged(`[${new Date().toISOString()}] ${note}\n`),
    append: (chunk: Uint8Array) => logged(chunk).pipe(Effect.zipRight(Ref.update(tail, bytes => chunk.length >= 16_384
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
    ))),
  }
})
