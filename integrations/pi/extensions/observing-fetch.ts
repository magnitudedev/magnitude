import { Either, Schema } from "effect"

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

const observeResponse = async (
  response: Response,
  request: ProgressRequest,
  signal: AbortSignal,
): Promise<void> => {
  if (response.body === null) {
    request.finish()
    return
  }
  const reader = response.body.getReader()
  const abort = () => {
    request.fail()
    void reader.cancel().catch(() => {})
  }
  if (signal.aborted) {
    abort()
    reader.releaseLock()
    return
  }
  signal.addEventListener("abort", abort, { once: true })
  const decoder = new TextDecoder()
  const parser = new SseDataParser()
  const observe = (value: unknown) => {
    const observation = decodeMagnitudeObservation(value)
    if (observation !== undefined) request.observe(observation)
  }
  try {
    while (true) {
      const result = await reader.read()
      if (result.done) break
      parser.push(decoder.decode(result.value, { stream: true }), observe)
    }
    parser.push(decoder.decode(), observe)
    parser.finish(observe)
    request.finish()
  } catch {
    request.fail()
  } finally {
    signal.removeEventListener("abort", abort)
    reader.releaseLock()
  }
}

export const makeObservingFetch = (
  fetchImplementation: typeof fetch,
  begin: () => ProgressRequest,
): typeof fetch => {
  const observingFetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const request = new Request(input, init)
    request.headers.set(MAGNITUDE_PROGRESS_HEADER, "true")
    const progress = begin()
    try {
      const response = await fetchImplementation(request)
      try {
        void observeResponse(response.clone(), progress, request.signal).catch(() => progress.fail())
      } catch {
        progress.fail()
      }
      return response
    } catch (error) {
      progress.fail()
      throw error
    }
  }
  return Object.assign(observingFetch, {
    preconnect: (...args: Parameters<typeof fetch.preconnect>) => fetchImplementation.preconnect?.(...args),
  })
}
