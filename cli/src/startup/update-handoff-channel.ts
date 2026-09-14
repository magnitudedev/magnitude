import { Effect, Schema } from "effect"

export class UpdateHandoffChannelFailed extends Schema.TaggedError<UpdateHandoffChannelFailed>()("UpdateHandoffChannelFailed", {}) {}

/** Read the request before acknowledging it, so the helper can acquire native exclusion first. */
export const readUpdateHandoff = <A, I>(input: AsyncIterable<Uint8Array>, schema: Schema.Schema<A, I>) => Effect.gen(function* () {
  const iterator = input[Symbol.asyncIterator]()
  const next = Effect.tryPromise({ try: () => iterator.next(), catch: () => new UpdateHandoffChannelFailed() })
  let buffer = Buffer.alloc(0)
  while (true) {
    const chunk = yield* next
    if (chunk.done) return yield* new UpdateHandoffChannelFailed()
    buffer = Buffer.concat([buffer, chunk.value])
    if (buffer.length > 32_768) return yield* new UpdateHandoffChannelFailed()
    const newline = buffer.indexOf(10)
    if (newline < 0) continue
    if (newline !== buffer.length - 1) return yield* new UpdateHandoffChannelFailed()
    const text = yield* Effect.try({
      try: () => new TextDecoder("utf-8", { fatal: true }).decode(buffer.subarray(0, newline)),
      catch: () => new UpdateHandoffChannelFailed(),
    })
    const request = yield* Schema.decodeUnknown(Schema.parseJson(schema))(text, { onExcessProperty: "error" }).pipe(
      Effect.mapError(() => new UpdateHandoffChannelFailed()),
    )
    return { request, awaitOwnerExit: next.pipe(Effect.flatMap(value => value.done ? Effect.void : new UpdateHandoffChannelFailed())) }
  }
})
