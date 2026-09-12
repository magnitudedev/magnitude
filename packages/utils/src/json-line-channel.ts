import type { Duplex } from "node:stream"
import { StringDecoder } from "node:string_decoder"
import { Effect, Schema, Stream } from "effect"

export class JsonLineChannelFailed extends Schema.TaggedError<JsonLineChannelFailed>()("JsonLineChannelFailed", {
  message: Schema.String,
}) {}
const maximumFrameBytes = 64 * 1024

/** One subscriber owns reads; input is bounded before JSON decoding. */
export const receiveJsonLines = <A, I>(socket: Duplex, schema: Schema.Schema<A, I>) => Stream.asyncScoped<A, JsonLineChannelFailed>(emit =>
  Effect.acquireRelease(Effect.sync(() => {
    const decoder = new StringDecoder("utf8")
    let pending = ""
    let failed = false
    const fail = (message: string) => {
      if (failed) return
      failed = true
      void emit.fail(new JsonLineChannelFailed({ message }))
      socket.destroy()
    }
    const onData = (chunk: Buffer) => {
      if (failed) return
      pending += decoder.write(chunk)
      for (;;) {
        const newline = pending.indexOf("\n")
        if (newline < 0) break
        const frame = pending.slice(0, newline)
        pending = pending.slice(newline + 1)
        if (Buffer.byteLength(frame) > maximumFrameBytes) return fail("Control frame exceeds 64 KiB")
        const decoded = Schema.decodeUnknownEither(Schema.parseJson(schema))(frame)
        if (decoded._tag === "Left") return fail("Invalid control frame")
        void emit.single(decoded.right)
      }
      if (Buffer.byteLength(pending) > maximumFrameBytes) fail("Unterminated control frame exceeds 64 KiB")
    }
    const onError = (error: Error) => fail(error.message)
    const onEnd = () => {
      pending += decoder.end()
      if (pending.length) fail("Incomplete control frame at EOF")
      else void emit.end()
    }
    socket.on("data", onData)
    socket.on("error", onError)
    socket.on("end", onEnd)
    return () => { socket.off("data", onData); socket.off("error", onError); socket.off("end", onEnd) }
  }), cleanup => Effect.sync(cleanup)),
)

export const sendJsonLine = <A, I>(socket: Duplex, schema: Schema.Schema<A, I>, value: NoInfer<A>) => Effect.gen(function* () {
  const encoded = yield* Schema.encode(Schema.parseJson(schema))(value).pipe(
    Effect.mapError(() => new JsonLineChannelFailed({ message: "Cannot encode control frame" })),
  )
  if (Buffer.byteLength(encoded) > maximumFrameBytes) return yield* new JsonLineChannelFailed({ message: "Control frame exceeds 64 KiB" })
  yield* Effect.async<void, JsonLineChannelFailed>(resume => {
    if (socket.destroyed || !socket.writable) return resume(Effect.fail(new JsonLineChannelFailed({ message: "Control channel is closed" })))
    const onError = (error: Error) => resume(Effect.fail(new JsonLineChannelFailed({ message: error.message })))
    socket.on("error", onError)
    socket.write(`${encoded}\n`, error => {
      socket.off("error", onError)
      resume(error ? Effect.fail(new JsonLineChannelFailed({ message: error.message })) : Effect.void)
    })
    return Effect.sync(() => socket.off("error", onError))
  }).pipe(Effect.timeoutFail({ duration: "2 seconds", onTimeout: () => new JsonLineChannelFailed({ message: "Control write timed out" }) }))
})
