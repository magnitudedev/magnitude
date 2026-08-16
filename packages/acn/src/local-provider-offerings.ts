import { Context, Effect, Layer, Option, Schema, Stream } from "effect"
import {
  LocalModelMutationFailed,
  type LocalInferenceError,
  type LocalProviderOffering,
  type ModelServingConfiguration,
  type ModelPackageEntry,
  servableModelBundlePackageIds,
  PRIMARY_SLOT_ID,
  SECONDARY_SLOT_ID,
  ProviderModelCatalogEntrySchema,
  type ProviderModelCatalogEntry,
  type ModelServingConfigurationId,
} from "@magnitudedev/acn-protocol"
import type { ProviderModelId } from "@magnitudedev/sdk"
import { PROVIDER_ID as LOCAL_PROVIDER_ID } from "@magnitudedev/icn/provider"
import { LocalModelPackages } from "./local-model-packages"
import {
  LocalModelConfigurationResolver,
  type ResolvedLocalModelConfiguration,
} from "./local-model-configuration-resolver"
import { makeObservedState } from "./mirrored-state"
import { resolveBundlePresentation } from "./local-model-presentation"
export { localProviderModelId, localCatalogProviderModelId } from "./local-provider-model-id"
import { localCatalogProviderModelId, localProviderModelId } from "./local-provider-model-id"

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
  offerings: readonly Pick<LocalProviderOffering, "providerModelId" | "configuration">[],
  entries: ReadonlyMap<ModelPackageEntry["package"]["id"], ModelPackageEntry>,
): ProviderOfferingPackageEvidence => [...offerings]
  .sort((left, right) => left.providerModelId.localeCompare(right.providerModelId))
  .map((offering) => ({
    providerModelId: offering.providerModelId,
    configurationId: offering.configuration.id,
    packages: servableModelBundlePackageIds(offering.configuration.bundle).map((packageId) => {
      const entry = entries.get(packageId)
      return {
        packageId,
        installed: entry?.localState._tag === "Installed",
        inspection: entry?.inspection._tag ?? "Pending",
      }
    }).sort((left, right) => left.packageId.localeCompare(right.packageId)),
  }))

export const sameProviderOfferingPackageEvidence = (
  left: ProviderOfferingPackageEvidence,
  right: ProviderOfferingPackageEvidence,
): boolean => left.length === right.length && left.every((offering, index) => {
  const other = right[index]
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

export interface LocalProviderOfferingsState {
  readonly packageEvidence: Option.Option<ProviderOfferingPackageEvidence>
  readonly entries: readonly ProviderModelCatalogEntry[]
  readonly failure: Option.Option<LocalInferenceError>
}

const failure = (operation: string, error: unknown) =>
  new LocalModelMutationFailed({
    code: operation,
    message: error instanceof Error ? error.message : String(error),
    retryable: true,
  })

export interface LocalProviderOfferingsApi {
  readonly ready: Effect.Effect<boolean>
  readonly list: Effect.Effect<readonly LocalProviderOffering[], LocalInferenceError>
  readonly changes: Stream.Stream<void>
  readonly catalog: Effect.Effect<readonly ProviderModelCatalogEntry[], LocalInferenceError>
  readonly state: Effect.Effect<LocalProviderOfferingsState>
  readonly catalogChanges: Stream.Stream<void>
  readonly resolve: (
    providerModelId: ProviderModelId,
  ) => Effect.Effect<LocalProviderOffering, LocalInferenceError>
}

export class LocalProviderOfferings extends Context.Tag("LocalProviderOfferings")<
  LocalProviderOfferings,
  LocalProviderOfferingsApi
>() {}

const providerAvailability = (
  installed: boolean,
  inspectable: boolean,
  assessment: ResolvedLocalModelConfiguration["assessment"],
): ProviderModelCatalogEntry["availability"] => {
  if (!installed) {
    return { _tag: "Disabled", reason: "installation_unavailable" }
  }
  if (!inspectable) {
    return { _tag: "Disabled", reason: "provider_unavailable" }
  }
  switch (assessment._tag) {
    case "Fits":
      return { _tag: "Available" }
    case "DoesNotFit":
      return { _tag: "Disabled", reason: "insufficient_resources" }
    case "Incompatible":
      return { _tag: "Disabled", reason: "incompatible_runtime" }
    case "Assessing":
    case "Failed":
      return { _tag: "Disabled", reason: "provider_unavailable" }
  }
}

export const LocalProviderOfferingsLive: Layer.Layer<
  LocalProviderOfferings,
  never,
  LocalModelConfigurationResolver | LocalModelPackages
> = Layer.scoped(LocalProviderOfferings, Effect.gen(function* () {
  const resolver = yield* LocalModelConfigurationResolver
  const packages = yield* LocalModelPackages

  const readyOfferingsFrom = (
    resolved: readonly ResolvedLocalModelConfiguration[],
  ) => resolved.flatMap((resolution) => resolution.targetInspection._tag === "Inspected"
    ? [{
      resolution,
      offering: {
        providerModelId: Option.match(resolution.catalogModel, {
          onNone: () => localProviderModelId(resolution.servingConfiguration.id),
          onSome: localCatalogProviderModelId,
        }),
        configuration: resolution.servingConfiguration,
        capabilities: resolution.targetInspection.capabilities,
      } satisfies LocalProviderOffering,
    }]
    : [])

  const list: LocalProviderOfferingsApi["list"] = Effect.gen(function* () {
    const resolved = [...(yield* resolver.get).values()]
    return readyOfferingsFrom(resolved).map(({ offering }) => offering)
  }).pipe(
    Effect.mapError((error) => error instanceof LocalModelMutationFailed
      ? error
      : failure("read_local_provider_offerings_failed", error)),
  )

  const changes = resolver.changes

  const observed = yield* makeObservedState<LocalProviderOfferingsState>({
    packageEvidence: Option.none(),
    entries: [],
    failure: Option.none(),
  })
  const entriesEquivalent = Schema.equivalence(Schema.Array(ProviderModelCatalogEntrySchema))
  const stateEquivalent = (
    left: LocalProviderOfferingsState,
    right: LocalProviderOfferingsState,
  ): boolean => Option.match(left.packageEvidence, {
    onNone: () => Option.isNone(right.packageEvidence),
    onSome: (evidence) => Option.exists(right.packageEvidence, (other) =>
      sameProviderOfferingPackageEvidence(evidence, other)),
  }) && entriesEquivalent(left.entries, right.entries)
    && Option.getOrUndefined(left.failure)?.message === Option.getOrUndefined(right.failure)?.message

  const compute = Effect.gen(function* () {
    const resolved = [...(yield* resolver.get).values()]
    const packageSnapshot = yield* packages.snapshot
    const packageEntries = new Map(
      packageSnapshot.state.entries.map((entry) => [entry.package.id, entry]),
    )
    const configured = readyOfferingsFrom(resolved)
    const offerings = configured.map(({ offering }) => offering)
    const packageEvidence = providerOfferingPackageEvidence(offerings, packageEntries)
    const bundleEntries = offerings.map(({ configuration }) =>
      servableModelBundlePackageIds(configuration.bundle).map((id) => packageEntries.get(id)))
    const installedBundles = bundleEntries.map((entries) =>
      entries.every((entry) => entry?.localState._tag === "Installed"))
    const inspectable = bundleEntries.map((entries, index) => installedBundles[index]
      && entries.every((entry) => entry?.inspection._tag === "Inspected"))
    const entries = configured.map(({ offering, resolution }, index): ProviderModelCatalogEntry => {
      const { bundle, profile } = offering.configuration
      const installed = installedBundles[index] ?? false
      const bundleInspectable = inspectable[index] ?? false
      const assessment = resolution.assessment
      const curated = Option.getOrUndefined(resolution.catalogModel)
      const presentation = resolveBundlePresentation(bundle, curated && {
        displayName: curated.displayName,
        variantLabel: curated.variantLabel,
        description: curated.description,
        license: curated.license,
      })
      return {
        providerId: LOCAL_PROVIDER_ID,
        providerModelId: offering.providerModelId,
        modelFamilyId: Option.none(),
        displayName: presentation.displayName,
        variantLabel: Option.some(presentation.variantLabel),
        supportedSlots: [PRIMARY_SLOT_ID, SECONDARY_SLOT_ID],
        contextWindow: profile.contextLength,
        maxOutputTokens: profile.contextLength,
        memory: bundleInspectable && assessment._tag === "Fits"
          ? Option.some(assessment.assessment.memory)
          : bundleInspectable && assessment._tag === "DoesNotFit"
            ? Option.some(assessment.memory)
            : Option.none(),
        capabilities: offering.capabilities,
        availability: providerAvailability(installed, bundleInspectable, assessment),
        pricing: Option.none(),
      }
    })
    return {
      entries,
      packageEvidence,
    }
  })

  const publishCurrent: Effect.Effect<void, LocalInferenceError> = compute.pipe(
    Effect.flatMap(({ entries, packageEvidence }) => observed.setIfChanged({
      packageEvidence: Option.some(packageEvidence),
      entries,
      failure: Option.none(),
    }, stateEquivalent)),
  )

  const project = publishCurrent.pipe(
    Effect.catchAll((error) => observed.get.pipe(Effect.flatMap(({ state }) =>
      observed.setIfChanged(
        { ...state, failure: Option.some(error) },
        stateEquivalent,
      ).pipe(Effect.asVoid)))),
    Effect.catchAllCause((cause) => Effect.logWarning(
      "Unable to project local provider offerings",
    ).pipe(Effect.annotateLogs({ cause: String(cause) }))),
  )
  yield* Stream.make(undefined).pipe(
    Stream.concat(changes.pipe(Stream.debounce("25 millis"))),
    Stream.runForEach(() => project),
    Effect.forkScoped,
  )

  return LocalProviderOfferings.of({
    ready: resolver.settled,
    list,
    changes,
    catalog: observed.get.pipe(Effect.flatMap(({ state }) => Option.match(state.failure, {
      onNone: () => Effect.succeed(state.entries),
      onSome: Effect.fail,
    }))),
    state: observed.get.pipe(Effect.map(({ state }) => state)),
    catalogChanges: observed.changes.pipe(Stream.map(() => undefined)),
    resolve: (providerModelId) => list.pipe(Effect.flatMap((offerings) => {
      const offering = offerings.find((candidate) => candidate.providerModelId === providerModelId)
      return offering
        ? Effect.succeed(offering)
        : Effect.fail(new LocalModelMutationFailed({
            code: "local_provider_offering_not_found",
            message: `Local provider offering ${providerModelId} was not found`,
            retryable: false,
          }))
    })),
  })
}))
