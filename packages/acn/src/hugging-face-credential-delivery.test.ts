import { Effect, Layer, Option, Redacted, Stream, SubscriptionRef } from "effect"
import { describe, expect, it } from "vitest"
import { IcnHuggingFaceCredential } from "@magnitudedev/icn"
import { AcnChanges } from "./changes"
import { HuggingFaceAccount } from "./hugging-face-account"
import { HuggingFaceCredentialDeliveryLive } from "./hugging-face-credential-delivery"

type Session = { readonly username: Option.Option<string>; readonly token: Option.Option<string> }

/** An account whose token and signed-in username follow `session`, and a change registry recording every poke. */
const fixtures = (session: SubscriptionRef.SubscriptionRef<Session>, pokes: string[]) => Layer.mergeAll(
  Layer.succeed(HuggingFaceAccount, {
    account: SubscriptionRef.get(session).pipe(Effect.map(({ username }) => Option.match(username, {
      onNone: () => ({ _tag: "SignedOut" as const }),
      onSome: (name) => ({ _tag: "SignedIn" as const, username: name }),
    }))),
    token: SubscriptionRef.get(session).pipe(Effect.map(({ token }) => Option.map(token, Redacted.make))),
    tokenChanges: session.changes.pipe(
      Stream.map(({ token }) => Option.map(token, Redacted.make)),
      Stream.changesWith(Option.getEquivalence((a, b) => Redacted.value(a) === Redacted.value(b))),
    ),
  } as never),
  Layer.succeed(AcnChanges, { publish: ({ operation }: { readonly operation: string }) => Effect.sync(() => { pokes.push(operation) }) } as never),
)

describe("Hugging Face credential delivery to ICN", () => {
  it("hands ICN the current token at start and every change after, without restarting it", async () => {
    const delivered: Option.Option<string>[] = []
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const session = yield* SubscriptionRef.make<Session>({ username: Option.some("tom"), token: Option.some("hf_oauth_one") })
      const icn = Layer.succeed(IcnHuggingFaceCredential, {
        configure: (value) => Effect.sync(() => { delivered.push(Option.map(value, Redacted.value)) }),
      })
      yield* Layer.build(HuggingFaceCredentialDeliveryLive.pipe(Layer.provide(Layer.merge(fixtures(session, []), icn))))
      const atStart = [...delivered]
      yield* Effect.sleep("10 millis")
      yield* SubscriptionRef.set(session, { username: Option.some("tom"), token: Option.some("hf_oauth_two") })
      yield* Effect.sleep("10 millis")
      yield* SubscriptionRef.set(session, { username: Option.none(), token: Option.none() })
      yield* Effect.sleep("20 millis")
      return atStart
    })))
    expect(result).toEqual([Option.some("hf_oauth_one")])
    expect(delivered).toEqual([Option.some("hf_oauth_one"), Option.some("hf_oauth_two"), Option.none()])
  })

  it("has clients refetch what they read from the Hub once ICN serves another account or none, not on renewal", async () => {
    const pokes: string[] = []
    const order: string[] = []
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const session = yield* SubscriptionRef.make<Session>({ username: Option.some("tom"), token: Option.some("hf_oauth_one") })
      const icn = Layer.succeed(IcnHuggingFaceCredential, {
        configure: (value) => Effect.sync(() => { order.push(Option.isSome(value) ? "configure token" : "configure none") }),
      })
      yield* Layer.build(HuggingFaceCredentialDeliveryLive.pipe(Layer.provide(Layer.merge(fixtures(session, pokes), icn))))
      yield* SubscriptionRef.set(session, { username: Option.some("tom"), token: Option.some("hf_oauth_renewed") })
      yield* Effect.sleep("10 millis")
      expect(pokes).toEqual([])
      yield* SubscriptionRef.set(session, { username: Option.none(), token: Option.none() })
      yield* Effect.sleep("10 millis")
    })))
    expect(pokes).toEqual(["SearchHuggingFaceModels", "InspectModelCandidate", "StreamModelCandidateInspection", "ListDrafterCandidates"])
    expect(order.at(-1)).toBe("configure none")
  })

  it("keeps ACN running when ICN refuses the credential", async () => {
    let attempts = 0
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const session = yield* SubscriptionRef.make<Session>({ username: Option.some("tom"), token: Option.some("hf_oauth_one") })
      const icn = Layer.succeed(IcnHuggingFaceCredential, {
        configure: () => Effect.suspend(() => { attempts += 1; return Effect.fail({ _tag: "GeneratedClientTransportError" } as never) }),
      })
      yield* Layer.build(HuggingFaceCredentialDeliveryLive.pipe(Layer.provide(Layer.merge(fixtures(session, []), icn))))
    })))
    expect(attempts).toBe(5)
  })
})
