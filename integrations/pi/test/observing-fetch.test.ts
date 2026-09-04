import { afterEach, describe, expect, it, vi } from "vitest"
import { Effect, Exit, Scope } from "effect"
import { MAGNITUDE_PROGRESS_HEADER, makeObservingFetch, SseDataParser } from "../extensions/observing-fetch"

const settle = () => new Promise((resolve) => setTimeout(resolve, 0))
const scopes: Scope.CloseableScope[] = []
const scope = () => { const value = Effect.runSync(Scope.make()); scopes.push(value); return value }
afterEach(async () => { for (const value of scopes.splice(0)) await Effect.runPromise(Scope.close(value, Exit.void)) })

describe("SseDataParser", () => {
  it("handles fragmented CRLF, comments, multiline data, malformed JSON, and a final unterminated event", () => {
    const values: unknown[] = []
    const parser = new SseDataParser()
    parser.push(": keepalive\r\nda", (value) => values.push(value))
    parser.push("ta: {\"progress\":\r\ndata: {\"phase\":\"queued\"}}\r\n\r\ndata: nope\n\n", (value) => values.push(value))
    parser.push("data: {\"progress\":{\"phase\":\"generating\"}}", (value) => values.push(value))
    parser.finish((value) => values.push(value))
    expect(values).toEqual([
      { progress: { phase: "queued" } },
      { progress: { phase: "generating" } },
    ])
  })
})

describe("makeObservingFetch", () => {
  it("does not prevent inference when progress startup throws", async () => {
    const upstream = Object.assign(vi.fn(async () => new Response("unmodified")), { preconnect: vi.fn() }) as typeof fetch
    const response = await makeObservingFetch(upstream, () => { throw new Error("presentation failed") }, scope())("http://localhost")
    expect(await response.text()).toBe("unmodified")
    expect(upstream).toHaveBeenCalledOnce()
  })

  it("cancels its clone when the owning extension scope closes", async () => {
    const cancelled = vi.fn()
    const failed = vi.fn()
    const owned = scope()
    const upstream = Object.assign(vi.fn(async () => new Response(new ReadableStream({ start(controller) { controller.enqueue(new TextEncoder().encode('data: {}\n\n')) }, cancel: cancelled }))), { preconnect: vi.fn() }) as typeof fetch
    const response = await makeObservingFetch(upstream, () => Effect.succeed({ observe: () => Effect.void, finish: Effect.void, fail: Effect.sync(failed) }), owned)("http://localhost")
    await settle()
    await Effect.runPromise(Scope.close(owned, Exit.void))
    await response.body!.cancel()
    expect(cancelled).toHaveBeenCalledOnce()
    expect(failed).toHaveBeenCalledOnce()
  })
  it("opts into progress while returning the original response body untouched", async () => {
    const events = [
      'data: {"progress":{"phase":"model_loading","fraction":0.47}}\n\n',
      'data: {"progress":{"phase":"prefill","completed_tokens":14020,"total_tokens":14300,"cached_tokens":13200}}\n\n',
      'data: {"timings":{"prompt_ms":500,"time_to_first_token_ms":3400,"predicted_n":17,"predicted_ms":247.38,"predicted_per_second":68.72}}\n\n',
      "data: [DONE]\n\n",
    ].join("")
    const observe = vi.fn()
    const finish = vi.fn()
    const fail = vi.fn()
    const upstream = Object.assign(vi.fn(async (request: Request) => {
      expect(request.headers.get(MAGNITUDE_PROGRESS_HEADER)).toBe("true")
      return new Response(events, { headers: { "content-type": "text/event-stream" } })
    }), { preconnect: vi.fn() }) as typeof fetch
    const wrapped = makeObservingFetch(upstream, () => Effect.succeed({ observe: (value) => Effect.sync(() => observe(value)), finish: Effect.sync(finish), fail: Effect.sync(fail) }), scope())

    const response = await wrapped("http://127.0.0.1/v1/chat/completions", {
      headers: { "Magnitude-Include-Progress": "false", "x-user": "preserved" },
    })
    expect(await response.text()).toBe(events)
    await settle()
    expect(observe).toHaveBeenCalledWith({ progress: { phase: "model_loading", fraction: 0.47 } })
    expect(observe).toHaveBeenCalledWith({
      progress: { phase: "prefill", completed_tokens: 14_020, total_tokens: 14_300, cached_tokens: 13_200 },
    })
    expect(observe).toHaveBeenCalledWith({
      timings: {
        prompt_ms: 500,
        time_to_first_token_ms: 3_400,
        predicted_n: 17,
        predicted_ms: 247.38,
        predicted_per_second: 68.72,
      },
    })
    expect(finish).toHaveBeenCalledOnce()
    expect(fail).not.toHaveBeenCalled()
  })

  it("clears progress when the request or observational clone fails", async () => {
    const fail = vi.fn()
    const upstream = Object.assign(vi.fn(async () => { throw new Error("offline") }), { preconnect: vi.fn() }) as typeof fetch
    await expect(makeObservingFetch(upstream, () => Effect.succeed({ observe: () => Effect.void, finish: Effect.void, fail: Effect.sync(fail) }), scope())("http://x"))
      .rejects.toThrow("offline")
    expect(fail).toHaveBeenCalledOnce()
  })

  it("cancels the observational branch and clears status when Pi aborts a request", async () => {
    const controller = new AbortController()
    const fail = vi.fn()
    const cancelled = vi.fn()
    const upstream = Object.assign(vi.fn(async () => new Response(new ReadableStream({
      start(stream) {
        stream.enqueue(new TextEncoder().encode('data: {"progress":{"phase":"generating"}}\n\n'))
      },
      cancel: cancelled,
    }))), { preconnect: vi.fn() }) as typeof fetch
    const response = await makeObservingFetch(upstream, () => Effect.succeed({
      observe: () => Effect.void,
      finish: Effect.void,
      fail: Effect.sync(fail),
    }), scope())("http://x", { signal: controller.signal })

    controller.abort()
    // A cloned response tees the source. Both branches must be cancelled before
    // the underlying stream receives its cancellation signal.
    await response.body?.cancel()
    await settle()

    expect(fail).toHaveBeenCalled()
    expect(cancelled).toHaveBeenCalled()
  })
})
