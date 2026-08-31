import { Context, Effect, Layer, Option, Stream } from "effect"
import {
  localModelProviderModelId,
  localModelServingState,
  LocalModelMutationFailed,
  PRIMARY_SLOT_ID,
  SECONDARY_SLOT_ID,
  type LocalInferenceError,
  type LocalModel,
  type LocalModelServingState,
  type LocalModelsState,
  type LocalProviderOffering,
  type ProviderModelCatalogEntry,
} from "@magnitudedev/acn-protocol"
import type { ProviderModelId } from "@magnitudedev/sdk"
import { PROVIDER_ID as LOCAL_PROVIDER_ID } from "@magnitudedev/icn/provider"
import { LocalModels } from "./local-models"

export interface LocalProviderOfferingsApi {
  readonly ready: Effect.Effect<boolean>
  readonly list: Effect.Effect<readonly LocalProviderOffering[]>
  readonly changes: Stream.Stream<void>
  readonly catalog: Effect.Effect<readonly ProviderModelCatalogEntry[]>
  readonly catalogChanges: Stream.Stream<void>
  readonly resolve: (providerModelId: ProviderModelId) => Effect.Effect<LocalProviderOffering, LocalInferenceError>
}
export class LocalProviderOfferings extends Context.Tag("LocalProviderOfferings")<LocalProviderOfferings, LocalProviderOfferingsApi>() {}

type AssessedServingState = Extract<LocalModelServingState, {
  readonly _tag: "Assessed"
}>

const providerAvailability = (
  model: LocalModel,
  serving: AssessedServingState,
): ProviderModelCatalogEntry["availability"] => {
  if (serving.assessment._tag === "DoesNotFit") {
    return { _tag: "Disabled", reason: "insufficient_resources" }
  }
  if (serving.assessment._tag === "Incompatible") {
    return { _tag: "Disabled", reason: "incompatible_runtime" }
  }
  if (Option.isSome(localModelProviderModelId(model))) return { _tag: "Available" }
  return model._tag === "Catalog"
    && (model.acquisitionState._tag === "NotInstalled"
      || model.acquisitionState._tag === "Installing"
      || model.acquisitionState._tag === "InstallFailed")
    ? { _tag: "Disabled", reason: "installation_unavailable" }
    : { _tag: "Disabled", reason: "provider_unavailable" }
}

export const localProviderOfferingsReady = (state: LocalModelsState): boolean =>
  state.reconciliationComplete
  && state.models.every((model) => Option.match(localModelServingState(model), {
    onNone: () => true,
    onSome: (serving) => serving._tag !== "Assessing",
  }))

export const projectLocalProviderOfferings = (models: readonly LocalModel[]) => {
  const assessedModels = models.flatMap((model) => Option.match(localModelServingState(model), {
    onNone: () => [],
    onSome: (serving) => serving._tag === "Assessed"
      ? [{ model, serving }]
      : [],
  }))
  const selected = assessedModels.filter(({ model, serving }) =>
    serving.assessment._tag === "Fits" && Option.isSome(localModelProviderModelId(model)))
  return {
    offerings: selected.map(({ model, serving }): LocalProviderOffering => ({
      providerModelId: model.modelId,
      profile: serving.assessment.profile,
      capabilities: serving.capabilities,
    })),
    entries: assessedModels.map(({ model, serving }): ProviderModelCatalogEntry => {
      return {
        providerId: LOCAL_PROVIDER_ID,
        providerModelId: model.modelId,
        modelFamilyId: Option.none(),
        displayName: model.presentation.displayName,
        variantLabel: Option.some(model.presentation.variantLabel),
        supportedSlots: [PRIMARY_SLOT_ID, SECONDARY_SLOT_ID],
        contextWindow: serving.assessment.profile.contextLength,
        maxOutputTokens: serving.assessment.profile.contextLength,
        memory: serving.assessment._tag === "Fits" ? Option.some(serving.assessment.memory.domains)
          : serving.assessment._tag === "DoesNotFit" ? Option.some(serving.assessment.memoryDomains)
            : Option.none(),
        capabilities: serving.capabilities,
        availability: providerAvailability(model, serving),
        pricing: Option.none(),
      }
    }),
  }
}

export const LocalProviderOfferingsLive: Layer.Layer<LocalProviderOfferings, never, LocalModels> =
  Layer.effect(LocalProviderOfferings, Effect.gen(function* () {
    const models = yield* LocalModels
    const compute = models.state.pipe(Effect.map((state) => projectLocalProviderOfferings(state.models)))
    const list = compute.pipe(Effect.map((state) => state.offerings))
    return LocalProviderOfferings.of({
      ready: models.state.pipe(Effect.map(localProviderOfferingsReady)),
      list,
      changes: models.changes,
      catalog: compute.pipe(Effect.map((state) => state.entries)),
      catalogChanges: models.changes,
      resolve: (providerModelId) => list.pipe(Effect.flatMap((offerings) => {
        const offering = offerings.find((candidate) => candidate.providerModelId === providerModelId)
        return offering === undefined
          ? Effect.fail(new LocalModelMutationFailed({ code: "local_provider_offering_not_found",
              message: `Local provider offering ${providerModelId} was not found`, retryable: false }))
          : Effect.succeed(offering)
      })),
    })
  }))
