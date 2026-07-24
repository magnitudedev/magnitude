import { Context, Effect, Layer, Option, Schema, Stream } from "effect"
import {
  PRIMARY_SLOT_ID,
  ProviderModelCatalogEntrySchema,
  SECONDARY_SLOT_ID,
  type LocalInferenceError,
  type ProviderModelCatalogEntry,
  modelOfferingTargetPackageIds,
} from "@magnitudedev/protocol"
import { IcnCatalog, IcnHardware } from "@magnitudedev/icn"
import { PROVIDER_ID as LOCAL_PROVIDER_ID } from "@magnitudedev/icn/provider"
import { LocalModelEvaluations } from "./local-model-evaluations"
import { LocalModelPackages } from "./local-model-packages"
import { LocalProviderOfferings } from "./local-provider-offerings"
import { recommendableModelFromIcn } from "./local-model-icn-adapter"
import { makeObservedState } from "./mirrored-state"

export interface LocalProviderOfferingProjectionState {
  readonly entries: readonly ProviderModelCatalogEntry[]
  readonly failure: Option.Option<LocalInferenceError>
}

export interface LocalProviderOfferingProjectionApi {
  readonly list: Effect.Effect<readonly ProviderModelCatalogEntry[], LocalInferenceError>
  readonly state: Effect.Effect<LocalProviderOfferingProjectionState>
  readonly changes: Stream.Stream<void>
}

export class LocalProviderOfferingProjection extends Context.Tag("LocalProviderOfferingProjection")<
  LocalProviderOfferingProjection,
  LocalProviderOfferingProjectionApi
>() {}

export const LocalProviderOfferingProjectionLive: Layer.Layer<
  LocalProviderOfferingProjection,
  never,
  IcnCatalog | IcnHardware | LocalModelEvaluations | LocalModelPackages | LocalProviderOfferings
> = Layer.scoped(LocalProviderOfferingProjection, Effect.gen(function* () {
  const catalog = yield* IcnCatalog
  const evaluations = yield* LocalModelEvaluations
  const hardware = yield* IcnHardware
  const packages = yield* LocalModelPackages
  const offerings = yield* LocalProviderOfferings

  const observed = yield* makeObservedState<LocalProviderOfferingProjectionState>({
    entries: [],
    failure: Option.none(),
  })
  const entriesEquivalent = Schema.equivalence(Schema.Array(ProviderModelCatalogEntrySchema))
  const equivalent = (
    left: LocalProviderOfferingProjectionState,
    right: LocalProviderOfferingProjectionState,
  ): boolean =>
    entriesEquivalent(left.entries, right.entries)
    && Option.getOrUndefined(left.failure)?.message === Option.getOrUndefined(right.failure)?.message
  const compute = Effect.gen(function* () {
    const recommendableModels = (yield* Effect.forEach(
      (yield* catalog.get).state.models,
      (model) => recommendableModelFromIcn(model).pipe(Effect.option),
    )).flatMap(Option.toArray)
    const curatedNames = new Map(recommendableModels.map((model) => [model.targetId, model.displayName]))
    const installedIds = yield* packages.installedPackageIds
    const configured = yield* offerings.list
    const installed = configured.map((offering) =>
      modelOfferingTargetPackageIds(offering.configuration.target).every((packageId) => installedIds.has(packageId)))
    const assessmentRequests = configured.flatMap((offering, index) => installed[index]
      ? [{
          target: offering.configuration.target,
          profiles: [offering.configuration.profile],
        }]
      : [])
    const assessed = yield* evaluations.assessMany(assessmentRequests)
    let assessmentIndex = 0
    const entries = configured.map((offering, index) => {
      const { target, profile } = offering.configuration
      const isInstalled = installed[index] ?? false
      const result = isInstalled ? assessed[assessmentIndex++] : undefined
      const assessment = result?._tag === "Assessed"
        ? result.assessments[0]
        : undefined
      const targetPackage = target._tag === "Package"
        ? target.package
        : target.target
      const sourceName = targetPackage.source._tag === "HuggingFace"
        ? targetPackage.source.repository.split("/").at(-1) ?? targetPackage.source.repository
        : targetPackage.files[0]?.path.split("/").at(-1) ?? targetPackage.id
      return {
        providerId: LOCAL_PROVIDER_ID,
        providerModelId: offering.providerModelId,
        modelFamilyId: Option.none(),
        displayName: curatedNames.get(offering.modelId) ?? (target._tag === "Package"
          ? `${sourceName} ${targetPackage.properties.quantization}`
          : `${sourceName} + speculative draft`),
        supportedSlots: [PRIMARY_SLOT_ID, SECONDARY_SLOT_ID],
        contextWindow: profile.contextLength,
        maxOutputTokens: Math.min(32_768, profile.contextLength),
        runtimeMemoryBytes: assessment?._tag === "Fits"
          ? Option.some(assessment.assessment.memory.reduce(
              (total, domain) => total + domain.requiredBytes,
              0,
            ))
          : Option.none(),
        capabilities: offering.capabilities,
        availability: !isInstalled
          ? { _tag: "Disabled" as const, reason: "installation_unavailable" as const }
          : assessment?._tag === "Fits"
            ? { _tag: "Available" as const }
            : assessment?._tag === "DoesNotFit"
              ? { _tag: "Disabled" as const, reason: "insufficient_resources" as const }
              : { _tag: "Disabled" as const, reason: "incompatible_runtime" as const },
        pricing: Option.none(),
      }
    })
    return entries
  })
  const project = compute.pipe(
    Effect.flatMap((entries) => observed.setIfChanged({
      entries,
      failure: Option.none(),
    }, equivalent)),
    Effect.catchAll((error) => observed.get.pipe(
      Effect.flatMap(({ state }) => observed.setIfChanged({
        entries: state.entries,
        failure: Option.some(error),
      }, equivalent)),
    )),
    Effect.catchAllCause((cause) =>
    Effect.logWarning("Unable to project local provider offerings").pipe(
      Effect.annotateLogs({ cause: String(cause) }),
    )),
  )

  yield* project
  yield* Stream.mergeAll([
    offerings.changes,
    catalog.changes.pipe(Stream.map(() => undefined)),
    packages.changes.pipe(Stream.map(() => undefined)),
    hardware.changes.pipe(Stream.map(() => undefined)),
  ], { concurrency: "unbounded" }).pipe(
    Stream.runForEach(() => project),
    Effect.forkScoped,
  )

  return LocalProviderOfferingProjection.of({
    state: observed.get.pipe(Effect.map(({ state }) => state)),
    list: observed.get.pipe(Effect.flatMap(({ state }) => Option.match(state.failure, {
      onNone: () => Effect.succeed(state.entries),
      onSome: Effect.fail,
    }))),
    changes: observed.changes.pipe(Stream.map(() => undefined)),
  })
}))
