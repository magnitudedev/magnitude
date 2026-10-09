import { Effect } from "effect"
import { describe, expect, it } from "vitest"
import { finishWithBody, makeInferenceActivity } from "./inference-activity"

const run = <A>(effect: Effect.Effect<A>) => Effect.runPromise(effect)

describe("inference activity", () => {
  it("counts a streamed request until its body is fully read", async () => {
    const activity = await run(makeInferenceActivity)
    const finish = await run(activity.begin)
    expect(await run(activity.count)).toBe(1)
    const stream = new ReadableStream<Uint8Array>({ start: controller => { controller.enqueue(new TextEncoder().encode("token")); controller.close() } })
    const response = finishWithBody(new Response(stream, { status: 200, headers: { "content-type": "text/event-stream" } }), finish)
    expect(response.headers.get("content-type")).toBe("text/event-stream")
    expect(await run(activity.count)).toBe(1)
    expect(await response.text()).toBe("token")
    expect(await run(activity.count)).toBe(0)
  })
  it("stops counting when the client cancels the body", async () => {
    const activity = await run(makeInferenceActivity)
    const response = finishWithBody(new Response(new ReadableStream({ pull: () => new Promise(() => {}) })), await run(activity.begin))
    await response.body!.cancel()
    expect(await run(activity.count)).toBe(0)
  })
  it("finishes a bodiless response at once and never counts below zero", async () => {
    const activity = await run(makeInferenceActivity)
    const finish = await run(activity.begin)
    finishWithBody(new Response(null, { status: 204 }), finish)
    finish()
    expect(await run(activity.count)).toBe(0)
  })
  it("counts a held connection for its scope", async () => {
    const activity = await run(makeInferenceActivity)
    await run(Effect.scoped(activity.hold.pipe(Effect.zipRight(activity.count), Effect.tap(count => Effect.sync(() => expect(count).toBe(1))))))
    expect(await run(activity.count)).toBe(0)
  })
})
