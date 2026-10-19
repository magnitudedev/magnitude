import { Effect, Layer, Option, Redacted, Ref, Schedule, Stream } from "effect"
import { Models } from "@magnitudedev/acn-protocol"
import { IcnHuggingFaceCredential } from "@magnitudedev/icn"
import { AcnChanges } from "./changes"
import { HuggingFaceAccount } from "./hugging-face-account"

/** What clients cache from the Hub through ICN, which reads differently for another account or none. */
const hubReadOperations = [
  Models.searchHuggingFaceModels._tag,
  Models.inspectModelCandidate._tag,
  Models.streamModelCandidateInspection._tag,
  Models.listDrafterCandidates._tag,
] as const

/**
 * Hands ICN the signed-in Hugging Face token, or none, once ICN is up and again whenever it changes,
 * so ICN applies it without restarting. A delivery ICN refuses is retried briefly and logged; ICN
 * then stays anonymous until the next change. Once ICN has a different account's token, or none,
 * clients refetch what they read from the Hub, so a gated model reads as the new account sees it; a
 * renewed token for the same account changes nothing they show.
 */
export const HuggingFaceCredentialDeliveryLive: Layer.Layer<
  never,
  never,
  HuggingFaceAccount | IcnHuggingFaceCredential | AcnChanges
> = Layer.scopedDiscard(Effect.gen(function* () {
  const account = yield* HuggingFaceAccount
  const icn = yield* IcnHuggingFaceCredential
  const changes = yield* AcnChanges
  const signedInAs = account.account.pipe(Effect.map((state) => state._tag === "SignedIn" ? Option.some(state.username) : Option.none<string>()))
  const shownAs = yield* Ref.make(yield* signedInAs)
  const refetchHubReads = signedInAs.pipe(
    Effect.flatMap((current) => Ref.getAndSet(shownAs, current).pipe(
      Effect.flatMap((previous) => Option.getEquivalence<string>((a, b) => a === b)(previous, current)
        ? Effect.void
        : Effect.forEach(hubReadOperations, (operation) => changes.publish({ operation }), { discard: true })),
    )),
  )
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
      Effect.flatMap((last) => same(last, token) ? Effect.void : Ref.set(delivered, token).pipe(Effect.zipRight(deliver(token)), Effect.zipRight(refetchHubReads))),
    )),
    Effect.forkScoped,
  )
}))
