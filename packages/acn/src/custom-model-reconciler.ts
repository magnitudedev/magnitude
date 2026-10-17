import { Cause, Context, Effect, Layer, Option, Ref, Stream } from "effect"
import {
  LocalModelMutationFailed,
  Models,
  type CustomFormModelId,
  type ModelId,
  type SpeculativeMethod,
} from "@magnitudedev/acn-protocol"
import { IcnCandidates, IcnLocalModels } from "@magnitudedev/icn"
import { PROVIDER_ID as LOCAL_PROVIDER_ID } from "@magnitudedev/icn/provider"
import type { CustomModelsConfig } from "@magnitudedev/storage"
import type * as Native from "@magnitudedev/icn-protocol/schemas"
import { AcnChanges } from "./changes"
import { CustomModels, localModelsConfiguration } from "./custom-models"
import { drafterReferenceKey, productDrafterReference } from "./drafter-references"
import { ModelLocations } from "./model-locations"
import { ModelSlotController } from "./model-slot-controller"

/**
 * Delivers ACN's custom-model choices to ICN as one whole snapshot. Commands `push` it so ICN reflects
 * a choice before they return.
 */
export interface CustomModelReconcilerApi {
  readonly push: Effect.Effect<void, LocalModelMutationFailed>
}
export class CustomModelReconciler extends Context.Tag("CustomModelReconciler")<
  CustomModelReconciler,
  CustomModelReconcilerApi
>() {}

/** Models a change makes disappear from selection: forgotten declarations and newly hidden models. */
export const disappearedCustomModels = (
  previous: CustomModelsConfig,
  next: CustomModelsConfig,
): readonly ModelId[] => {
  const forgotten = (Object.keys(previous.declarations) as CustomFormModelId[])
    .filter((modelId) => !(modelId in next.declarations))
  const hidden = next.hidden.filter((modelId) => !previous.hidden.includes(modelId))
  return [...new Set([...forgotten, ...hidden])]
}

const pushFailed = () => new LocalModelMutationFailed({
  code: "local_models_configuration_failed",
  message: "The inference service did not accept your model choices. Try again.",
  retryable: true,
})

export const CustomModelReconcilerLive: Layer.Layer<
  CustomModelReconciler,
  never,
  CustomModels | ModelLocations | IcnLocalModels | IcnCandidates | ModelSlotController | AcnChanges
> = Layer.scoped(CustomModelReconciler, Effect.gen(function* () {
  const customModels = yield* CustomModels
  const locations = yield* ModelLocations
  const localModels = yield* IcnLocalModels
  const candidates = yield* IcnCandidates
  const slots = yield* ModelSlotController
  const changes = yield* AcnChanges
  const lock = yield* Effect.makeSemaphore(1)

  // A chosen drafter is delivered with its persisted method. ICN's candidates only validate it: a drafter
  // the engine now binds with another method has its persisted method corrected. An added model still
  // downloading has no files to list from, so its declared repository file is asked.
  const validateMethods = (config: CustomModelsConfig) => Effect.forEach(
    Object.entries(config.drafters),
    ([id, choice]) => Effect.gen(function* () {
      if (choice._tag !== "Reference") return
      const modelId = id as CustomFormModelId
      const key = drafterReferenceKey(choice.reference)
      const bound = (request: unknown) => candidates.drafters(request as Native.DrafterCandidatesRequest).pipe(
        Effect.map(({ candidates: listed }) => listed),
        Effect.orElseSucceed((): readonly Native.DrafterCandidate[] => []),
        Effect.map((listed) => Option.fromNullable(listed.find((candidate) => {
          const reference = productDrafterReference(candidate.reference)
          return candidate.verdict._tag === "Bound" && Option.isSome(reference) && drafterReferenceKey(reference.value) === key
        })).pipe(Option.flatMap((candidate) => candidate.verdict._tag === "Bound"
          ? Option.some(candidate.verdict.method as SpeculativeMethod)
          : Option.none()))),
      )
      const declaration = config.declarations[modelId]
      const method = yield* bound({ _tag: "Model", modelId, other: Option.none() }).pipe(
        Effect.flatMap((method) => Option.isSome(method) || declaration?.origin._tag !== "Added"
          ? Effect.succeed(method)
          : bound({
              _tag: "Candidate",
              candidate: {
                source: { _tag: "HuggingFace", repository: declaration.origin.repository, commit: declaration.origin.commit },
                target: [...declaration.target],
                projector: Option.none(),
                drafter: Option.none(),
              },
              other: Option.none(),
            })),
      )
      if (Option.isNone(method) || method.value._tag === choice.method._tag) return
      const corrected = method.value
      yield* customModels.modify((current) => {
        const latest = current.drafters[modelId]
        const stale = latest?._tag === "Reference" && drafterReferenceKey(latest.reference) === key
          && latest.method._tag !== corrected._tag
        return Effect.succeed([undefined, stale
          ? { ...current, drafters: { ...current.drafters, [modelId]: { ...latest, method: corrected } } }
          : current] as const)
      })
    }),
    { discard: true },
  )

  const push = lock.withPermits(1)(Effect.gen(function* () {
    const config = yield* customModels.get
    const configuration = localModelsConfiguration(config, yield* locations.resolve)
    yield* localModels.configure(configuration).pipe(Effect.mapError(pushFailed))
  }))

  const previous = yield* Ref.make(yield* customModels.get)
  const reconcile = (next: CustomModelsConfig) => Effect.gen(function* () {
    const disappeared = disappearedCustomModels(yield* Ref.get(previous), next)
    if (disappeared.length > 0) {
      const snapshot = (yield* slots.state).slots
      for (const slot of [snapshot.primary, snapshot.secondary]) {
        if (slot._tag !== "Unassigned" && slot.selection.providerId === LOCAL_PROVIDER_ID
          && disappeared.some((modelId) => modelId === slot.selection.providerModelId)) {
          yield* slots.updateModelSlot(slot.slotId, Option.none())
        }
      }
    }
    yield* Ref.set(previous, next)
    yield* push
    yield* changes.publish({ operation: Models.listModelLocations._tag })
    yield* validateMethods(next)
  }).pipe(Effect.catchAllCause((cause) => Effect.logError("Unable to reconcile custom models").pipe(
    Effect.annotateLogs({ cause: Cause.pretty(cause).slice(0, 2_000) }),
  )))

  // The first emission is the current choices. Every later change is compared with the last one seen.
  yield* customModels.changes.pipe(Stream.runForEach(reconcile), Effect.forkScoped)

  return CustomModelReconciler.of({ push })
}))
