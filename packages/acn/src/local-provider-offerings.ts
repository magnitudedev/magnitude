import { Context, Effect, Layer, Option, Stream } from "effect"
import {
  LocalModelMutationFailed,
  type LocalInferenceError,
  type LocalProviderOffering,
  type ModelServingConfiguration,
  type ModelPackageEntry,
  servableModelBundlePackageIds,
  PRIMARY_SLOT_ID,
  SECONDARY_SLOT_ID,
  type ProviderModelCatalogEntry,
} from "@magnitudedev/acn-protocol"
import type { ProviderModelId } from "@magnitudedev/sdk"
import { PROVIDER_ID as LOCAL_PROVIDER_ID } from "@magnitudedev/icn/provider"
import { LocalModelPackages } from "./local-model-packages"
import {
  LocalModelConfigurationResolver,
  type ResolvedLocalModelConfiguration,
} from "./local-model-configuration-resolver"
import { resolveBundlePresentation } from "./local-model-presentation"
export { localCatalogProviderModelId } from "./local-provider-model-id"
import { localCatalogProviderModelId } from "./local-provider-model-id"

export interface LocalProviderOfferingsState {
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

export interface ProjectedLocalProviderOfferings {
  readonly offerings: readonly LocalProviderOffering[]
  readonly entries: readonly ProviderModelCatalogEntry[]
}

const configuredOfferings = (
  resolved: readonly ResolvedLocalModelConfiguration[],
) => resolved.flatMap((resolution) => resolution.targetInspection._tag === "Inspected"
  && Option.isSome(resolution.catalogModel)
  ? [{
      resolution,
      offering: {
        providerModelId: localCatalogProviderModelId(resolution.catalogModel.value),
        configuration: resolution.servingConfiguration,
        capabilities: resolution.targetInspection.capabilities,
      } satisfies LocalProviderOffering,
    }]
  : [])

type PackageAvailabilityEvidence = readonly {
  readonly packageId: ModelPackageEntry["package"]["id"]
  readonly installed: boolean
  readonly inspectable: boolean
}[]

const packageAvailabilityEvidence = (
  entries: readonly ModelPackageEntry[],
): PackageAvailabilityEvidence => entries.map((entry) => ({
  packageId: entry.package.id,
  installed: entry.localState._tag === "Installed",
  inspectable: entry.inspection._tag === "Inspected",
}))

const samePackageAvailability = (
  left: PackageAvailabilityEvidence,
  right: PackageAvailabilityEvidence,
): boolean => left.length === right.length && left.every((entry, index) => {
  const other = right[index]
  return other !== undefined
    && entry.packageId === other.packageId
    && entry.installed === other.installed
    && entry.inspectable === other.inspectable
})

export const projectLocalProviderOfferings = (
  resolved: readonly ResolvedLocalModelConfiguration[],
  packageEntries: ReadonlyMap<ModelPackageEntry["package"]["id"], ModelPackageEntry>,
): ProjectedLocalProviderOfferings => {
  const configured = configuredOfferings(resolved)
  const offerings = configured.map(({ offering }) => offering)
  const entries = configured.map(({ offering, resolution }): ProviderModelCatalogEntry => {
    const bundleEntries = servableModelBundlePackageIds(offering.configuration.bundle)
      .map((id) => packageEntries.get(id))
    const installed = bundleEntries.every((entry) => entry?.localState._tag === "Installed")
    const inspectable = installed
      && bundleEntries.every((entry) => entry?.inspection._tag === "Inspected")
    const { bundle, profile } = offering.configuration
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
      memory: inspectable && assessment._tag === "Fits"
        ? Option.some(assessment.assessment.memory)
        : inspectable && assessment._tag === "DoesNotFit"
          ? Option.some(assessment.memory)
          : Option.none(),
      capabilities: offering.capabilities,
      availability: providerAvailability(installed, inspectable, assessment),
      pricing: Option.none(),
    }
  })
  return { offerings, entries }
}

export const LocalProviderOfferingsLive: Layer.Layer<
  LocalProviderOfferings,
  never,
  LocalModelConfigurationResolver | LocalModelPackages
> = Layer.effect(LocalProviderOfferings, Effect.gen(function* () {
  const resolver = yield* LocalModelConfigurationResolver
  const packages = yield* LocalModelPackages

  const list: LocalProviderOfferingsApi["list"] = resolver.get.pipe(
    Effect.map((resolved) => configuredOfferings([...resolved.values()]).map(({ offering }) => offering)),
    Effect.mapError((error) => error instanceof LocalModelMutationFailed
      ? error
      : failure("read_local_provider_offerings_failed", error)),
  )

  const changes = Stream.merge(
    resolver.changes,
    packages.changes.pipe(
      Stream.map((state) => packageAvailabilityEvidence(state.entries)),
      Stream.changesWith(samePackageAvailability),
      Stream.map(() => undefined),
    ),
  )

  const compute = Effect.gen(function* () {
    const resolved = [...(yield* resolver.get).values()]
    const packageState = yield* packages.state
    const packageEntries = new Map(
      packageState.entries.map((entry) => [entry.package.id, entry]),
    )
    return projectLocalProviderOfferings(resolved, packageEntries)
  })

  return LocalProviderOfferings.of({
    ready: resolver.settled,
    list,
    changes,
    catalog: compute.pipe(Effect.map(({ entries }) => entries)),
    state: compute.pipe(
      Effect.map(({ entries }): LocalProviderOfferingsState => ({
        entries,
        failure: Option.none(),
      })),
      Effect.catchAll((error) => Effect.succeed({
        entries: [],
        failure: Option.some(error),
      })),
    ),
    catalogChanges: changes,
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
