import { Effect } from "effect"

export interface FlushableOutput {
  readonly write: (
    chunk: string,
    callback: (error?: Error | null) => void,
  ) => unknown
}

export interface FlushProcessOutputOptions {
  readonly stdout?: FlushableOutput
  readonly stderr?: FlushableOutput
  readonly timeoutMs?: number
}

export const DEFAULT_PROCESS_OUTPUT_FLUSH_TIMEOUT_MS = 1_000
export const DEFAULT_BOUNDED_PROCESS_EXIT_GRACE_MS = 1_000

export function scheduleBoundedProcessExit(
  exitCode: number,
  graceMs = DEFAULT_BOUNDED_PROCESS_EXIT_GRACE_MS,
): Effect.Effect<void> {
  return Effect.sync(() => {
    process.exitCode = exitCode
    const timer = setTimeout(() => process.exit(exitCode), graceMs)
    timer.unref()
  })
}

export function flushProcessOutput(
  options: FlushProcessOutputOptions = {},
): Effect.Effect<void> {
  const timeoutMs = options.timeoutMs ?? DEFAULT_PROCESS_OUTPUT_FLUSH_TIMEOUT_MS
  const flush = (stream: FlushableOutput) => Effect.async<void>((resume) => {
    let settled = false
    const finish = () => {
      if (settled) return
      settled = true
      resume(Effect.void)
    }
    try {
      stream.write("", finish)
    } catch {
      finish()
    }
  }).pipe(Effect.timeoutOption(timeoutMs), Effect.asVoid)
  return Effect.all([
    flush(options.stdout ?? process.stdout),
    flush(options.stderr ?? process.stderr),
  ], { discard: true, concurrency: "unbounded" })
}
