import { Array, Context, Effect, Layer, Option, Order, Schema, Stream } from "effect"
import {
  LocalModelFamilySchema,
  Models,
  comparisonTokensPerSecond,
  localModelIntelligence,
  parseModelId,
  type CatalogBaseId,
  type CatalogLocalModel,
  type CountryCode,
  type LocalInferenceHardware as LocalInferenceHardwareState,
  type LocalModelsState,
  type LocalModelFamily,
  type RecommendedModel,
  type Recommendations as RecommendationsState,
} from "@magnitudedev/acn-protocol"
import type { ModelFamily as NativeModelFamily } from "@magnitudedev/icn-protocol/schemas"
import { AcnChanges } from "./changes"
import { LocalInferenceHardware } from "./local-inference-hardware"
import { LocalModelSources, type CatalogModelSource } from "./local-model-sources"
import { LocalModels } from "./local-models"

/** The decode speed a chosen rung should reach: a long agent conversation stays responsive. */
export const TARGET_TOKENS_PER_SECOND = 20
/** The decode speed below which onboarding recommends no model. */
export const ONBOARDING_FLOOR_TOKENS_PER_SECOND = 10

export interface RecommendationInputs {
  readonly models: LocalModelsState["models"]
  readonly catalog: readonly CatalogModelSource[]
  readonly families: readonly LocalModelFamily[]
  readonly hardware: LocalInferenceHardwareState
  readonly developerCountry: Option.Option<CountryCode>
}

interface Rung {
  readonly model: CatalogLocalModel
  readonly measured: boolean
  readonly fits: boolean
  readonly speed: Option.Option<number>
  readonly roomOnDisk: boolean
}

const decodeFamilies = Schema.decodeUnknownSync(Schema.Array(LocalModelFamilySchema))

/** The families Magnitude runs, as the release catalog declares them. */
export const projectModelFamilies = (families: readonly NativeModelFamily[]): readonly LocalModelFamily[] =>
  decodeFamilies(families)

/** The bytes a rung still has to download: none once it is on this computer. */
const remainingDownloadBytes = (model: CatalogLocalModel): number => {
  switch (model.acquisitionState._tag) {
    case "NotInstalled":
    case "InstallFailed":
      return model.storageBytes + (model.drafterWithheld
        ? 0
        : Option.match(model.catalogData.drafter, { onNone: () => 0, onSome: ({ sizeBytes }) => sizeBytes }))
    default:
      return 0
  }
}

const rung = (model: CatalogLocalModel, hardware: LocalInferenceHardwareState): Rung => {
  const serving = model.servingState
  const fits = serving._tag === "Assessed" && serving.assessment._tag === "Fits" ? serving.assessment : undefined
  return {
    model,
    measured: serving._tag !== "Assessing" && fits?.residency.performance._tag !== "Pending",
    fits: fits !== undefined,
    speed: fits === undefined
      ? Option.none()
      : comparisonTokensPerSecond(fits.residency.performance, fits.served.contextLength),
    roomOnDisk: Option.match(hardware.modelStorage, {
      onNone: () => true,
      onSome: ({ freeBytes }) => remainingDownloadBytes(model) <= freeBytes,
    }),
  }
}

const reaches = (rung: Rung, tokensPerSecond: number): boolean =>
  Option.exists(rung.speed, (speed) => speed >= tokensPerSecond)

/**
 * The rung chosen for this machine: the highest that fits, has room on disk and reaches the target speed; else the
 * fastest that fits.
 */
const chooseRung = (ladder: readonly Rung[]): Option.Option<Rung> => {
  const fitting = ladder.filter((candidate) => candidate.fits)
  const target = fitting.filter((candidate) => candidate.roomOnDisk && reaches(candidate, TARGET_TOKENS_PER_SECOND))
  if (target.length > 0) return Array.last(target)
  return Array.head(Array.sort(fitting, Order.mapInput(
    Order.reverse(Order.number),
    (candidate: Rung) => Option.getOrElse(candidate.speed, () => 0),
  )))
}

const byIntelligence = Order.mapInput(
  Order.reverse(Option.getOrder(Order.number)),
  (model: RecommendedModel) => model.intelligence,
)

/** Models whose chosen rung reaches the target first, then the rest with a fitting rung, then those with none. */
const standing = (model: RecommendedModel): number =>
  model.meetsTarget ? 0 : Option.isSome(model.chosen) ? 1 : 2

const recommendationOrder = Order.combine(
  Order.mapInput(Order.number, standing),
  byIntelligence,
)

/**
 * The tested models in the order they are recommended for this machine. A model's rungs are its catalog variants,
 * smallest to largest. Disabled and deprecated models are left out, except those of a family that is coming soon,
 * which are listed with no chosen rung. It is complete once every rung of a listed model has been measured.
 */
export const projectRecommendations = (inputs: RecommendationInputs): RecommendationsState => {
  const familyStatus = new Map(inputs.families.flatMap((family) =>
    family.architectures.map((architecture) => [architecture, family.status] as const)))
  const architectures = new Map(inputs.catalog.map((entry) =>
    [entry.id as string, entry.source.desired.metadata.architecture] as const))
  const comingSoon = (model: CatalogLocalModel): boolean =>
    familyStatus.get(architectures.get(model.modelId) ?? "") === "ComingSoon"
  const catalogModels = inputs.models.filter((model): model is CatalogLocalModel =>
    model._tag === "Catalog"
    && (model.catalogData.support._tag === "Supported" || comingSoon(model))
    && Option.match(inputs.developerCountry, {
      onNone: () => true,
      onSome: (country) => model.catalogData.developer.country === country,
    }))
  const ladders = Array.groupBy(catalogModels, (model) => {
    const parsed = parseModelId(model.modelId)
    return parsed._tag === "Catalog" ? parsed.baseId : model.modelId
  })
  let measured = 0
  let rungs = 0
  const models = Object.entries(ladders).flatMap(([base, variants]): readonly RecommendedModel[] => {
    const ladder = Array.sort(variants, Order.mapInput(Order.number, (model: CatalogLocalModel) => model.storageBytes))
      .map((model) => rung(model, inputs.hardware))
    if (!Array.isNonEmptyReadonlyArray(ladder)) return []
    const first = ladder[0].model
    if (!comingSoon(first)) {
      rungs += ladder.length
      measured += ladder.filter((candidate) => candidate.measured).length
    }
    const chosen = comingSoon(first) ? Option.none() : chooseRung(ladder)
    return [{
      base: base as CatalogBaseId,
      name: first.presentation.displayName,
      developer: first.catalogData.developer,
      intelligence: localModelIntelligence(first),
      chosen: Option.map(chosen, ({ model }) => model.modelId),
      rungs: Array.map(ladder, ({ model }) => model.modelId),
      meetsTarget: Option.exists(chosen, (candidate) => reaches(candidate, TARGET_TOKENS_PER_SECOND)),
      meetsOnboardingFloor: Option.exists(chosen, (candidate) => reaches(candidate, ONBOARDING_FLOOR_TOKENS_PER_SECOND)),
    }]
  })
  return { complete: measured === rungs, measured, rungs, models: Array.sort(models, recommendationOrder), families: inputs.families }
}

export interface RecommendationsApi {
  readonly get: (developerCountry: Option.Option<CountryCode>) => Effect.Effect<RecommendationsState>
}

export class Recommendations extends Context.Tag("Recommendations")<Recommendations, RecommendationsApi>() {}

export const RecommendationsLive: Layer.Layer<
  Recommendations,
  never,
  LocalModels | LocalModelSources | LocalInferenceHardware | AcnChanges
> = Layer.scoped(Recommendations, Effect.gen(function* () {
  const local = yield* LocalModels
  const sources = yield* LocalModelSources
  const hardware = yield* LocalInferenceHardware
  const changes = yield* AcnChanges
  yield* Stream.merge(local.changes, hardware.changes.pipe(Stream.map(() => undefined))).pipe(
    Stream.runForEach(() => changes.publish({ operation: Models.getRecommendations._tag })),
    Effect.forkScoped,
  )
  return Recommendations.of({
    get: (developerCountry) => Effect.all({ local: local.state, source: sources.state, hardware: hardware.state }).pipe(
      Effect.map(({ local, source, hardware }) => projectRecommendations({
        models: local.models,
        catalog: source.catalogModels,
        families: projectModelFamilies(source.families),
        hardware,
        developerCountry,
      })),
    ),
  })
}))
