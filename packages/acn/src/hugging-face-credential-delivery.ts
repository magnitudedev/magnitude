import { Effect, Layer, Option, Redacted, Ref, Schedule, Stream } from "effect"
import { IcnHuggingFaceCredential } from "@magnitudedev/icn"
import { HuggingFaceAccount } from "./hugging-face-account"

/**
 * Hands ICN the signed-in Hugging Face token, or none, once ICN is up and again whenever it changes,
 * so ICN applies it without restarting. A delivery ICN refuses is retried briefly and logged; ICN
 * then stays anonymous until the next change.
 */
export const HuggingFaceCredentialDeliveryLive: Layer.Layer<
  never,
  never,
  HuggingFaceAccount | IcnHuggingFaceCredential
> = Layer.scopedDiscard(Effect.gen(function* () {
  const account = yield* HuggingFaceAccount
  const icn = yield* IcnHuggingFaceCredential
  const deliver = (token: Effect.Effect.Success<typeof account.token>) => icn.configure(token).pipe(
    Effect.retry(Schedule.exponential("100 millis").pipe(Schedule.intersect(Schedule.recurs(4)))),
    Effect.catchAll((error) => Effect.logWarning("The inference service did not accept the Hugging Face sign-in; it stays anonymous").pipe(
      Effect.annotateLogs({ cause: error._tag }),
    )),
  )
  // The watch starts with the current token, so a change between the first delivery and the watch
  // is never lost; a token ICN already has is not sent again.
  const initial = yield* account.token
  yield* deliver(initial)
  const delivered = yield* Ref.make(initial)
  const same = Option.getEquivalence<Redacted.Redacted<string>>((a, b) => Redacted.value(a) === Redacted.value(b))
  yield* account.tokenChanges.pipe(
    Stream.runForEach((token) => Ref.get(delivered).pipe(
      Effect.flatMap((last) => same(last, token) ? Effect.void : Ref.set(delivered, token).pipe(Effect.zipRight(deliver(token)))),
    )),
    Effect.forkScoped,
  )
}))
