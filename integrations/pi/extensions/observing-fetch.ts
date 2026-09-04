import { Effect, Either, Schema, Scope } from "effect"

import type { ProgressRequest } from "./progress"
import { decodeMagnitudeObservation } from "./protocol"

export const MAGNITUDE_PROGRESS_HEADER = "Magnitude-Include-Progress"
const JsonDocumentSchema = Schema.parseJson(Schema.Unknown)

export class SseDataParser {
  #buffer = ""
  #dataLines: string[] = []

  push(fragment: string, observe: (value: unknown) => void): void {
    this.#buffer += fragment
    while (true) {
      const newline = this.#buffer.indexOf("\n")
      if (newline < 0) return
      let line = this.#buffer.slice(0, newline)
      this.#buffer = this.#buffer.slice(newline + 1)
      if (line.endsWith("\r")) line = line.slice(0, -1)
      this.#line(line, observe)
    }
  }

  finish(observe: (value: unknown) => void): void {
    if (this.#buffer.length > 0) this.#line(this.#buffer.replace(/\r$/, ""), observe)
    this.#buffer = ""
    this.#dispatch(observe)
  }

  #line(line: string, observe: (value: unknown) => void): void {
    if (line === "") {
      this.#dispatch(observe)
      return
    }
    if (line.startsWith(":")) return
    const colon = line.indexOf(":")
    const field = colon < 0 ? line : line.slice(0, colon)
    if (field !== "data") return
    const raw = colon < 0 ? "" : line.slice(colon + 1)
    this.#dataLines.push(raw.startsWith(" ") ? raw.slice(1) : raw)
  }

  #dispatch(observe: (value: unknown) => void): void {
    if (this.#dataLines.length === 0) return
    const data = this.#dataLines.join("\n")
    this.#dataLines = []
    if (data === "[DONE]") return
    const decoded = Schema.decodeUnknownEither(JsonDocumentSchema)(data)
    // The provider parser remains authoritative; malformed observational data is ignored.
    if (Either.isRight(decoded)) observe(decoded.right)
  }
}

const observeResponse = (
  response: Response,
  request: ProgressRequest,
  signal: AbortSignal,
): Effect.Effect<void> => Effect.scoped(Effect.gen(function* () {
  if (!response.ok || response.body === null) { yield* request.fail; return }
  const reader = yield* Effect.acquireRelease(Effect.sync(() => response.body!.getReader()), (reader) =>
    Effect.promise(() => reader.cancel()).pipe(Effect.interruptible, Effect.timeout("100 millis"), Effect.ignore,
      Effect.ensuring(Effect.sync(() => reader.releaseLock()))))
  if (signal.aborted) { yield* request.fail; return }
  const decoder = new TextDecoder()
  const parser = new SseDataParser()
  const drain = Effect.gen(function* () {
    while (true) {
      const result = yield* Effect.tryPromise(() => reader.read())
      if (result.done) break
      const values: unknown[] = []
      parser.push(decoder.decode(result.value, { stream: true }), (value) => values.push(value))
      for (const value of values) {
        const observation = decodeMagnitudeObservation(value)
        if (observation) yield* request.observe(observation)
      }
    }
    // An unterminated final frame is not evidence of a completed observation.
    yield* request.finish
  })
  const aborted = Effect.async<void>((resume) => {
    const abort = () => resume(Effect.interrupt)
    signal.addEventListener("abort", abort, { once: true })
    if (signal.aborted) abort()
    return Effect.sync(() => signal.removeEventListener("abort", abort))
  })
  yield* Effect.raceFirst(drain, aborted).pipe(Effect.onError(() => request.fail))
})).pipe(Effect.catchAllCause(() => Effect.void))

export const makeObservingFetch = (
  fetchImplementation: typeof fetch,
  begin: () => Effect.Effect<ProgressRequest>,
  scope: Scope.Scope,
): typeof fetch => {
  const observingFetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const request = new Request(input, init)
    request.headers.set(MAGNITUDE_PROGRESS_HEADER, "true")
    const progress = Effect.runSync(Effect.suspend(begin).pipe(Effect.catchAllCause(() => Effect.succeed({
      observe: () => Effect.void, finish: Effect.void, fail: Effect.void,
    }))))
    try {
      const response = await fetchImplementation(request)
      try {
        const observation = observeResponse(response.clone(), progress, request.signal)
        Effect.runFork(Effect.forkIn(observation, scope))
      } catch {
        Effect.runSync(progress.fail.pipe(Effect.catchAllCause(() => Effect.void)))
      }
      return response
    } catch (error) {
      Effect.runSync(progress.fail.pipe(Effect.catchAllCause(() => Effect.void)))
      throw error
    }
  }
  return Object.assign(observingFetch, {
    preconnect: (...args: Parameters<typeof fetch.preconnect>) => fetchImplementation.preconnect?.(...args),
  })
}
