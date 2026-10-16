import { Cause, Context, Effect, Layer, Option, Ref, Stream } from "effect"
import {
  LocalModelMutationFailed,
  Models,
  type CustomFormModelId,
} from "@magnitudedev/acn-protocol"
import { IcnCandidates, IcnLocalModels } from "@magnitudedev/icn"
import { PROVIDER_ID as LOCAL_PROVIDER_ID } from "@magnitudedev/icn/provider"
import type { CustomModelsConfig } from "@magnitudedev/storage"
import type * as Native from "@magnitudedev/icn-protocol/schemas"
import { AcnChanges } from "./changes"
import { CustomModels, localModelsConfiguration } from "./custom-models"
import { DrafterMethods, drafterReferenceKey, productDrafterReference } from "./drafter-references"
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
): readonly CustomFormModelId[] => {
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
  CustomModels | ModelLocations | DrafterMethods | IcnLocalModels | IcnCandidates | ModelSlotController | AcnChanges
> = Layer.scoped(CustomModelReconciler, Effect.gen(function* () {
  const customModels = yield* CustomModels
  const locations = yield* ModelLocations
  const methods = yield* DrafterMethods
  const localModels = yield* IcnLocalModels
  const candidates = yield* IcnCandidates
  const slots = yield* ModelSlotController
  const changes = yield* AcnChanges
  const lock = yield* Effect.makeSemaphore(1)

  // A drafter's binding method is engine knowledge; after a restart it is relearned from ICN's candidates.
  const learnMethods = (config: CustomModelsConfig) => Effect.forEach(
    Object.entries(config.drafters),
    ([modelId, choice]) => Effect.gen(function* () {
      if (choice._tag !== "Reference" || Option.isSome(yield* methods.get(choice.reference))) return
      const listed = yield* candidates.drafters({ _tag: "Model", modelId, other: Option.none() } as unknown as Native.DrafterCandidatesRequest).pipe(Effect.orElseSucceed(() => []))
      const key = drafterReferenceKey(choice.reference)
      for (const candidate of listed) {
        if (candidate.verdict._tag !== "Bound") continue
        const reference = productDrafterReference(candidate.reference)
        if (Option.isSome(reference) && drafterReferenceKey(reference.value) === key) {
          yield* methods.remember(choice.reference, candidate.verdict.method)
        }
      }
    }),
    { discard: true },
  )

  const push = lock.withPermits(1)(Effect.gen(function* () {
    const config = yield* customModels.get
    yield* learnMethods(config)
    const configuration = localModelsConfiguration(config, yield* locations.resolve, yield* methods.snapshot)
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
  }).pipe(Effect.catchAllCause((cause) => Effect.logError("Unable to reconcile custom models").pipe(
    Effect.annotateLogs({ cause: Cause.pretty(cause).slice(0, 2_000) }),
  )))

  // The first emission is the current choices: the spawn snapshot omitted drafters whose method was
  // not yet known, and this push completes it. Every later change is compared with the last one seen.
  yield* customModels.changes.pipe(Stream.runForEach(reconcile), Effect.forkScoped)

  return CustomModelReconciler.of({ push })
}))
