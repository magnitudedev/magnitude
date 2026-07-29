import { Context, Effect, Layer, Option, Schema, Stream } from "effect"
import {
  PRIMARY_SLOT_ID,
  ProviderModelCatalogEntrySchema,
  SECONDARY_SLOT_ID,
  type LocalInferenceError,
  type LocalProviderOffering,
  type ModelPackageEntry,
  type ProviderModelCatalogEntry,
  modelOfferingTargetPackageIds,
} from "@magnitudedev/acn-protocol"
import { IcnCatalog, IcnHardware } from "@magnitudedev/icn"
import { PROVIDER_ID as LOCAL_PROVIDER_ID } from "@magnitudedev/icn/provider"
import { LocalModelEvaluations } from "./local-model-evaluations"
import { LocalModelPackages } from "./local-model-packages"
import { LocalProviderOfferings } from "./local-provider-offerings"
import { recommendableModelFromIcn } from "./local-model-icn-adapter"
import { makeObservedState } from "./mirrored-state"

export type ProviderOfferingPackageEvidence = readonly {
  readonly providerModelId: LocalProviderOffering["providerModelId"]
  readonly configurationId: LocalProviderOffering["configuration"]["id"]
  readonly packages: readonly {
    readonly packageId: ModelPackageEntry["package"]["id"]
    readonly installed: boolean
    readonly inspection: ModelPackageEntry["inspection"]["_tag"]
  }[]
}[]

export const providerOfferingPackageEvidence = (
  offerings: readonly LocalProviderOffering[],
  entries: ReadonlyMap<ModelPackageEntry["package"]["id"], ModelPackageEntry>,
): ProviderOfferingPackageEvidence => [...offerings]
  .sort((left, right) => left.providerModelId.localeCompare(right.providerModelId))
  .map((offering) => ({
    providerModelId: offering.providerModelId,
    configurationId: offering.configuration.id,
    packages: modelOfferingTargetPackageIds(offering.configuration.target)
      .map((packageId) => {
        const entry = entries.get(packageId)
        return {
          packageId,
          installed: entry?.localState._tag === "Installed",
          inspection: entry?.inspection._tag ?? "Pending",
        }
      })
      .sort((left, right) => left.packageId.localeCompare(right.packageId)),
  }))

export const sameProviderOfferingPackageEvidence = (
  left: ProviderOfferingPackageEvidence,
  right: ProviderOfferingPackageEvidence,
): boolean => left.length === right.length && left.every((offering, offeringIndex) => {
  const other = right[offeringIndex]
  return offering.providerModelId === other?.providerModelId
    && offering.configurationId === other.configurationId
    && offering.packages.length === other.packages.length
    && offering.packages.every((modelPackage, packageIndex) => {
      const otherPackage = other.packages[packageIndex]
      return modelPackage.packageId === otherPackage?.packageId
        && modelPackage.installed === otherPackage.installed
        && modelPackage.inspection === otherPackage.inspection
    })
})

export interface LocalProviderOfferingProjectionState {
  readonly packageEvidence: Option.Option<ProviderOfferingPackageEvidence>
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
    packageEvidence: Option.none(),
    entries: [],
    failure: Option.none(),
  })
  const entriesEquivalent = Schema.equivalence(Schema.Array(ProviderModelCatalogEntrySchema))
  const equivalent = (
    left: LocalProviderOfferingProjectionState,
    right: LocalProviderOfferingProjectionState,
  ): boolean =>
    Option.match(left.packageEvidence, {
      onNone: () => Option.isNone(right.packageEvidence),
      onSome: (leftEvidence) => Option.exists(
        right.packageEvidence,
        (rightEvidence) => sameProviderOfferingPackageEvidence(leftEvidence, rightEvidence),
      ),
    })
    && entriesEquivalent(left.entries, right.entries)
    && Option.getOrUndefined(left.failure)?.message === Option.getOrUndefined(right.failure)?.message
  const compute = Effect.gen(function* () {
    const recommendableModels = (yield* Effect.forEach(
      (yield* catalog.get).state.models,
      (model) => recommendableModelFromIcn(model).pipe(Effect.option),
    )).flatMap(Option.toArray)
    const curatedNames = new Map(recommendableModels.map((model) => [model.targetId, model.displayName]))
    const packageSnapshot = yield* packages.snapshot
    const packageEntries = new Map(
      packageSnapshot.state.entries.map((entry) => [entry.package.id, entry]),
    )
    const configured = yield* offerings.list
    const packageEvidence = providerOfferingPackageEvidence(configured, packageEntries)
    const targetEntries = configured.map((offering) =>
      modelOfferingTargetPackageIds(offering.configuration.target)
        .map((packageId) => packageEntries.get(packageId)))
    const installed = targetEntries.map((entries) =>
      entries.every((entry) => entry?.localState._tag === "Installed"))
    const inspectable = targetEntries.map((entries, index) => installed[index]
      && entries.every((entry) => entry?.inspection._tag === "Inspected"))
    const assessmentRequests = configured.flatMap((offering, index) => inspectable[index]
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
      const isInspectable = inspectable[index] ?? false
      const result = isInspectable ? assessed[assessmentIndex++] : undefined
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
        displayName: curatedNames.get(offering.targetId) ?? (target._tag === "Package"
          ? `${sourceName} ${targetPackage.properties.quantization}`
          : `${sourceName} + speculative draft`),
        supportedSlots: [PRIMARY_SLOT_ID, SECONDARY_SLOT_ID],
        contextWindow: profile.contextLength,
        maxOutputTokens: Math.min(32_768, profile.contextLength),
        memory: assessment?._tag === "Fits"
          ? Option.some(assessment.assessment.memory)
          : assessment?._tag === "DoesNotFit"
            ? Option.some(assessment.memory)
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
    return { entries, packageEvidence }
  })
  const publishCurrent: Effect.Effect<void, LocalInferenceError> = Effect.suspend(() => compute.pipe(
    Effect.flatMap(({ entries, packageEvidence }) => Effect.all({
      configured: offerings.list,
      packages: packages.snapshot,
    }).pipe(
      Effect.flatMap(({ configured, packages: latest }) => {
        const latestEntries = new Map(
          latest.state.entries.map((entry) => [entry.package.id, entry]),
        )
        const latestEvidence = providerOfferingPackageEvidence(configured, latestEntries)
        return sameProviderOfferingPackageEvidence(packageEvidence, latestEvidence)
          ? observed.setIfChanged({
            packageEvidence: Option.some(packageEvidence),
            entries,
            failure: Option.none(),
          }, equivalent)
          : publishCurrent
      }),
    )),
  ))
  const project = publishCurrent.pipe(
    Effect.catchAll((error) => observed.get.pipe(
      Effect.flatMap(({ state }) => observed.setIfChanged({
        packageEvidence: state.packageEvidence,
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
    hardware.fittingChanges.pipe(Stream.map(() => undefined)),
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
