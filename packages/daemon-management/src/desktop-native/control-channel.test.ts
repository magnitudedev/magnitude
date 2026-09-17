import { PassThrough } from "node:stream"
import { DesktopOwnerCommand } from "@magnitudedev/acn-protocol/desktop-control"
import { Effect, Fiber, Schema, Stream } from "effect"
import { describe, expect, it } from "vitest"
import { receiveJsonLines, sendJsonLine } from "@magnitudedev/utils/json-line-channel"

const observe = <A, I>(schema: Schema.Schema<A, I>, chunks: ReadonlyArray<Buffer | string>) => Effect.scoped(Effect.gen(function* () {
  const socket = new PassThrough()
  const reader = yield* receiveJsonLines(socket, schema).pipe(Stream.runCollect, Effect.forkScoped)
  yield* Effect.yieldNow()
  for (const chunk of chunks) socket.write(chunk)
  socket.end()
  return Array.from(yield* Fiber.join(reader))
}))

describe("inherited control framing", () => {
  it("handles split and batched messages", async () => {
    expect(await Effect.runPromise(observe(DesktopOwnerCommand, ['{"_tag":"Sta', 'rt"}\n{"_tag":"Shutdown"}\n']))).toEqual([{ _tag: "Start" }, { _tag: "Shutdown" }])
  })
  it("preserves UTF-8 across byte boundaries", async () => {
    const bytes = Buffer.from('"日本語"\n')
    expect(await Effect.runPromise(observe(Schema.String, Array.from(bytes, byte => Buffer.from([byte]))))).toEqual(["日本語"])
  })
  it.each(['{"_tag":"Other"}\n', '{"_tag":"Start"}', 'x'.repeat(65537), '"' + 'x'.repeat(65536) + '"\n'])("rejects malformed, incomplete, or oversized frames", async input => {
    const result = await Effect.runPromise(observe(DesktopOwnerCommand, [input]).pipe(Effect.either))
    expect(result._tag === "Left" && result.left._tag).toBe("JsonLineChannelFailed")
  })
  it("encodes and observes commands using the same schema", async () => {
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const socket = new PassThrough()
      const reader = yield* receiveJsonLines(socket, DesktopOwnerCommand).pipe(Stream.take(1), Stream.runCollect, Effect.forkScoped)
      yield* Effect.yieldNow()
      yield* sendJsonLine(socket, DesktopOwnerCommand, { _tag: "Shutdown" })
      return Array.from(yield* Fiber.join(reader))
    })))
    expect(result).toEqual([{ _tag: "Shutdown" }])
  })
})
