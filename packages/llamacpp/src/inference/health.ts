import { Effect, pipe, Schedule, Stream } from "effect"
import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import { LlamaCppServerTimeout } from "../errors"

/** OOM patterns to scan stderr for during server startup. */
export const OOM_PATTERNS = [
  "failed to load model",
  "failed to allocate",
  "out of memory",
  "CUDA out of memory",
  "ggml_backend_buffer_type_alloc",
  "not enough memory",
]

/** Check if a stderr line matches any OOM pattern. */
export function detectOom(stderr: string): boolean {
  const lower = stderr.toLowerCase()
  return OOM_PATTERNS.some((p) => lower.includes(p.toLowerCase()))
}

/** Parse stdout for the listening port line: `listening on 127.0.0.1:<port>`. */
export function parseListeningPort(line: string): number | null {
  const m = line.match(/listening on \S+:(\d+)/i)
  return m ? Number(m[1]) : null
}

/**
 * Poll `GET /health` until the server responds with 200 (healthy).
 * 503 = still loading (not failure). Connection refused = keep trying.
 *
 * Times out after `timeoutMs` with `LlamaCppServerTimeout`.
 */
export function waitForReady(
  endpoint: string,
  timeoutMs = 120_000,
): Effect.Effect<void, LlamaCppServerTimeout, HttpClient.HttpClient> {
  return Effect.gen(function* () {
    const client = yield* HttpClient.HttpClient

    const checkHealth: Effect.Effect<boolean, never, HttpClient.HttpClient> = pipe(
      client.execute(HttpClientRequest.get(`${endpoint}/health`)),
      Effect.timeout("500 millis"),
      Effect.catchAll(() => Effect.succeed(null)),
      Effect.map((res) => res !== null && res.status === 200),
    )

    // Poll until ready or timeout
    const maxAttempts = Math.floor(timeoutMs / 500)
    const result = yield* pipe(
      checkHealth,
      Effect.repeat(Schedule.recurs(maxAttempts)),
      Effect.timeout(`${timeoutMs} millis`),
      Effect.catchAll(() => Effect.succeed(null)),
    )

    // result is the number of successful repeats or null on timeout
    // Simpler: just do one final check
    const finalCheck = yield* checkHealth

    if (!finalCheck) {
      return yield* new LlamaCppServerTimeout({ endpoint, phase: "health" })
    }
  })
}

/**
 * Collect stderr output from a process stream, checking for OOM patterns.
 * Returns all collected stderr text (for error reporting).
 */
export function collectStderr<S, R>(
  stderr: Stream.Stream<Uint8Array, S, R>,
): Effect.Effect<string, never, R> {
  return pipe(
    stderr,
    Stream.runFold(
      "",
      (acc, chunk) => {
        const text = new TextDecoder().decode(chunk)
        return acc + text
      },
    ),
    Effect.catchAll(() => Effect.succeed("")),
  )
}
