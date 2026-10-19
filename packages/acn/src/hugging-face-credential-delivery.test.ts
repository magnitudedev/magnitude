import { Effect, Layer, Option, Redacted, Stream, SubscriptionRef } from "effect"
import { describe, expect, it } from "vitest"
import { IcnHuggingFaceCredential } from "@magnitudedev/icn"
import { HuggingFaceAccount } from "./hugging-face-account"
import { HuggingFaceCredentialDeliveryLive } from "./hugging-face-credential-delivery"

describe("Hugging Face credential delivery to ICN", () => {
  it("hands ICN the current token at start and every change after, without restarting it", async () => {
    const delivered: Option.Option<string>[] = []
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const token = yield* SubscriptionRef.make(Option.some(Redacted.make("hf_oauth_one")))
      const account = Layer.succeed(HuggingFaceAccount, {
        token: SubscriptionRef.get(token),
        tokenChanges: token.changes.pipe(Stream.changesWith(Option.getEquivalence((a, b) => Redacted.value(a) === Redacted.value(b)))),
      } as never)
      const icn = Layer.succeed(IcnHuggingFaceCredential, {
        configure: (value) => Effect.sync(() => { delivered.push(Option.map(value, Redacted.value)) }),
      })
      yield* Layer.build(HuggingFaceCredentialDeliveryLive.pipe(Layer.provide(Layer.merge(account, icn))))
      const atStart = [...delivered]
      yield* Effect.sleep("10 millis")
      yield* SubscriptionRef.set(token, Option.some(Redacted.make("hf_oauth_two")))
      yield* Effect.sleep("10 millis")
      yield* SubscriptionRef.set(token, Option.none())
      yield* Effect.sleep("20 millis")
      return atStart
    })))
    expect(result).toEqual([Option.some("hf_oauth_one")])
    expect(delivered).toEqual([Option.some("hf_oauth_one"), Option.some("hf_oauth_two"), Option.none()])
  })

  it("keeps ACN running when ICN refuses the credential", async () => {
    let attempts = 0
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const account = Layer.succeed(HuggingFaceAccount, {
        token: Effect.succeed(Option.some(Redacted.make("hf_oauth_one"))),
        tokenChanges: Stream.make(Option.some(Redacted.make("hf_oauth_one"))),
      } as never)
      const icn = Layer.succeed(IcnHuggingFaceCredential, {
        configure: () => Effect.suspend(() => { attempts += 1; return Effect.fail({ _tag: "GeneratedClientTransportError" } as never) }),
      })
      yield* Layer.build(HuggingFaceCredentialDeliveryLive.pipe(Layer.provide(Layer.merge(account, icn))))
    })))
    expect(attempts).toBe(5)
  })
})
