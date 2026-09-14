import { describe, expect, it } from "vitest"
import { Effect, Schema } from "effect"
import { readUpdateHandoff } from "./update-handoff-channel"

const Request = Schema.Struct({ version: Schema.String })
const input = (...chunks: string[]) => (async function* () { for (const chunk of chunks) yield Buffer.from(chunk) })()

describe("native update handoff channel", () => {
  it("admits a fragmented request before waiting for the owner to exit", async () => {
    let exited = false
    const stream = (async function* () {
      yield Buffer.from('{"version":')
      yield Buffer.from('"1.2.3"}\n')
      exited = true
    })()
    await Effect.runPromise(Effect.gen(function* () {
      const channel = yield* readUpdateHandoff(stream, Request)
      expect(channel.request).toEqual({ version: "1.2.3" })
      expect(exited).toBe(false)
      yield* channel.awaitOwnerExit
      expect(exited).toBe(true)
    }))
  })
  it.each([
    [], ['{"version":"1"}'], ['{"version":"1"}\nextra'],
    ['{"version":"1","extra":true}\n'], ["x".repeat(32_769)],
  ].map((chunks, index) => ({ chunks, index })))("rejects invalid request case $index", async ({ chunks }) => {
    const result = await Effect.runPromise(Effect.either(readUpdateHandoff(input(...chunks), Request)))
    expect(result._tag).toBe("Left")
  })
  it("rejects bytes arriving after admission instead of treating them as owner exit", async () => {
    const result = await Effect.runPromise(readUpdateHandoff(input('{"version":"1"}\n', "extra"), Request).pipe(
      Effect.flatMap(channel => channel.awaitOwnerExit), Effect.either,
    ))
    expect(result._tag).toBe("Left")
  })
})
