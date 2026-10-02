import { Effect, Option, Stream } from "effect"
import { describe, expect, it } from "vitest"
import { JsonLineChannelFailed } from "@magnitudedev/utils/json-line-channel"
import type { OwnerReply } from "@magnitudedev/acn-protocol"
import { AcnOwner, requestOwner, watchApplicationOwner } from "./application-owner"

const owner = (reply: Effect.Effect<OwnerReply, JsonLineChannelFailed>) => AcnOwner.of({
  awaitStart: Effect.void, awaitShutdown: Effect.never, reportHealth: () => Effect.void,
  ownerState: Stream.make({ owner: "Desktop" as const, capabilities: ["Quit" as const], updates: Option.none(), loginStartup: Option.none() }),
  request: () => reply,
})
const attempt = (reply: Effect.Effect<OwnerReply, JsonLineChannelFailed>) =>
  Effect.runPromise(requestOwner({ _tag: "Quit" }).pipe(Effect.provideService(AcnOwner, owner(reply)), Effect.either))

describe("owner requests", () => {
  it("succeeds when the owner acts", async () => {
    const result = await attempt(Effect.succeed({ _tag: "Done" }))
    expect(result._tag === "Right" && result.right).toEqual({})
  })
  it("reports a request the owner does not support", async () => {
    const result = await attempt(Effect.succeed({ _tag: "Unsupported" }))
    expect(result._tag === "Left" && result.left._tag).toBe("OwnerRequestUnsupported")
  })
  it("reports the owner's own failure message", async () => {
    const result = await attempt(Effect.succeed({ _tag: "Failed", message: "Stop the server first." }))
    expect(result._tag === "Left" && result.left._tag === "OwnerRequestFailed" && result.left.message).toBe("Stop the server first.")
  })
  it("reports an owner that is gone as unavailable", async () => {
    const result = await attempt(Effect.fail(new JsonLineChannelFailed({ message: "Desktop control channel closed" })))
    expect(result._tag === "Left" && result.left._tag).toBe("ApplicationOwnerUnavailable")
  })
  it("streams the owner's self-report", async () => {
    const states = await Effect.runPromise(Stream.runCollect(watchApplicationOwner.pipe(Stream.take(1))).pipe(Effect.provideService(AcnOwner, owner(Effect.never))))
    expect([...states].map(state => state.owner)).toEqual(["Desktop"])
  })
})
