import { describe, expect, it } from "vitest"
import { Clock, Duration, Effect, Layer, Stream } from "effect"
import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientResponse from "@effect/platform/HttpClientResponse"
import { executeHttpStream } from "../stream"
import { formatStreamFailureMessage } from "../../errors/classify"

describe("executeHttpStream", () => {
  it("preserves useful context when the response body stream fails", async () => {
    const encoder = new TextEncoder()
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode("data: "))
        controller.error(new TypeError("terminated"))
      },
    })

    const mockClient = HttpClient.make((req) =>
      Effect.succeed(
        HttpClientResponse.fromWeb(
          req,
          new Response(body, {
            status: 200,
            headers: {
              "content-type": "text/event-stream",
              "x-request-id": "request-123",
            },
          }),
        ),
      ),
    )
    const layer = Layer.succeed(HttpClient.HttpClient, mockClient)
    const clockLayer = Layer.succeed(Clock.Clock, {
      currentTimeMillis: () => Effect.succeed(0),
      currentTimeNanos: () => Effect.succeed(0n),
      sleep: (dur: unknown) => Effect.sleep(dur as Duration.Duration),
      unsafeCurrentTimeMillis: () => 0,
      unsafeCurrentTimeNanos: () => 0n,
      [Clock.ClockTypeId]: Clock.ClockTypeId,
    } as unknown as Clock.Clock)
    const mergedLayer = Layer.merge(layer, clockLayer)

    const program = Effect.gen(function* () {
      const result = yield* executeHttpStream({
        call: {
          provider: "https://app.magnitude.dev",
          model: "test-model",
          method: "POST",
          url: "https://app.magnitude.dev/api/v1/chat/completions",
        },
        body: {},
        auth: () => {},
        decodePayload: (raw) => Effect.succeed(raw),
      })

      expect(result.responseHeaders.get("x-request-id")).toBe("request-123")
      expect(result.response.requestId).toBe("request-123")

      return yield* Stream.runCollect(result.stream).pipe(Effect.either)
    })

    const result = await Effect.runPromise(program.pipe(Effect.provide(mergedLayer)))

    expect(result._tag).toBe("Left")
    if (result._tag !== "Left") return

    expect(result.left._tag).toBe("StreamOperationalFailure")
    if (result.left._tag !== "StreamOperationalFailure") return
    expect(result.left.reason._tag).toBe("BodyReadFailure")

    const message = formatStreamFailureMessage(result.left)
    expect(message).toContain("Model response stream failed operationally")
    expect(message).toContain("response: 200 POST https://app.magnitude.dev/api/v1/chat/completions")
    expect(message).toContain("effectReason=Decode")
    expect(message).toContain("cause=TypeError: terminated")
  })

  it("fails with StreamOperationalFailure/StallTimeout when an accepted response stops producing body bytes", async () => {
    const body = new ReadableStream<Uint8Array>({
      start() {
        // Intentionally leave the stream open without emitting bytes.
      },
    })

    const mockClient = HttpClient.make((req) =>
      Effect.succeed(
        HttpClientResponse.fromWeb(
          req,
          new Response(body, {
            status: 200,
            headers: { "content-type": "text/event-stream" },
          }),
        ),
      ),
    )
    const layer = Layer.succeed(HttpClient.HttpClient, mockClient)
    const clockLayer = Layer.succeed(Clock.Clock, {
      currentTimeMillis: () => Effect.succeed(0),
      currentTimeNanos: () => Effect.succeed(0n),
      sleep: (dur: unknown) => Effect.sleep(dur as Duration.Duration),
      unsafeCurrentTimeMillis: () => 0,
      unsafeCurrentTimeNanos: () => 0n,
      [Clock.ClockTypeId]: Clock.ClockTypeId,
    } as unknown as Clock.Clock)
    const mergedLayer = Layer.merge(layer, clockLayer)

    const program = Effect.gen(function* () {
      const result = yield* executeHttpStream({
        call: {
          provider: "https://app.magnitude.dev",
          model: "test-model",
          method: "POST",
          url: "https://app.magnitude.dev/api/v1/chat/completions",
        },
        body: {},
        auth: () => {},
        decodePayload: (raw) => Effect.succeed(raw),
        idleTimeoutMs: 5,
      })

      return yield* Stream.runCollect(result.stream).pipe(Effect.either)
    })

    const result = await Effect.runPromise(program.pipe(Effect.provide(mergedLayer)))

    expect(result._tag).toBe("Left")
    if (result._tag !== "Left") return

    expect(result.left._tag).toBe("StreamOperationalFailure")
    if (result.left._tag !== "StreamOperationalFailure") return
    expect(result.left.reason._tag).toBe("StallTimeout")
    if (result.left.reason._tag !== "StallTimeout") return
    expect(result.left.reason.timeoutMs).toBe(5)
    expect(result.left.reason.lastActivity._tag).toBe("ResponseAccepted")
  })
})
