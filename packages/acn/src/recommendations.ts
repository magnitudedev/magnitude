import { Array, Context, Effect, Layer, Option, Order, Schema, Stream } from "effect"
import {
  LocalModelFamilySchema,
  RecommendationsSchema,
  Models,
  comparisonTokensPerSecond,
  localModelCapabilities,
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
  type RoomShortfall,
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

/** The bytes a rung still has to download: its stored bytes, its catalog drafter among them unless withheld; none once it is on this computer. */
const remainingDownloadBytes = (model: CatalogLocalModel): number => {
  switch (model.acquisitionState._tag) {
    case "NotInstalled":
    case "InstallFailed":
      return model.storageBytes
    default:
      return 0
  }
}

const rung = (model: CatalogLocalModel, freeBytes: Option.Option<number>): Rung => {
  const serving = model.servingState
  const fits = serving._tag === "Assessed" && serving.assessment._tag === "Fits" ? serving.assessment : undefined
  return {
    model,
    measured: serving._tag !== "Assessing" && fits?.residency.performance._tag !== "Pending",
    fits: fits !== undefined,
    speed: fits === undefined
      ? Option.none()
      : comparisonTokensPerSecond(fits.residency.performance, fits.served.contextLength),
    roomOnDisk: Option.match(freeBytes, {
      onNone: () => true,
      onSome: (free) => remainingDownloadBytes(model) <= free,
    }),
  }
}

const reaches = (rung: Rung, tokensPerSecond: number): boolean =>
  Option.exists(rung.speed, (speed) => speed >= tokensPerSecond)

/** The share of the fastest rung's speed within which rungs decode alike; the inference service's recommendations hold the same share. */
const NEAR_FASTEST = 0.9

/**
 * The rung chosen for this machine, among those that fit and have room on disk: the highest that reaches the target
 * speed; else the highest within `NEAR_FASTEST` of the fastest, since a smaller rung a few percent faster is no
 * better choice; else, where no speed could be measured, the smallest. A rung still being measured is never chosen:
 * one whose measurement never ends would otherwise be offered though it may not load.
 */
const chooseRung = (ladder: readonly Rung[]): Option.Option<Rung> => {
  const fitting = ladder.filter((candidate) => candidate.fits && candidate.roomOnDisk && candidate.measured)
  const target = fitting.filter((candidate) => reaches(candidate, TARGET_TOKENS_PER_SECOND))
  if (target.length > 0) return Array.last(target)
  const fastest = Math.max(0, ...fitting.flatMap((candidate) => Option.toArray(candidate.speed)))
  return Array.last(fitting.filter((candidate) => reaches(candidate, fastest * NEAR_FASTEST))).pipe(
    Option.orElse(() => Array.head(fitting)),
  )
}

const byIntelligence = Order.mapInput(
  Order.reverse(Option.getOrder(Order.number)),
  (model: RecommendedModel) => model.intelligence,
)

/**
 * Models whose chosen rung reaches the target first, then those reaching the onboarding floor, then the rest with a
 * fitting rung, then those with none: a smarter model too slow to use never leads one that is usable.
 */
const standing = (model: RecommendedModel): number =>
  model.meetsTarget ? 0 : model.meetsOnboardingFloor ? 1 : Option.isSome(model.chosen) ? 2 : 3

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
    // Harnesses call tools, so a model known not to support them is never recommended.
    && !Option.exists(localModelCapabilities(model), (capabilities) => !capabilities.tools)
    && Option.match(inputs.developerCountry, {
      onNone: () => true,
      onSome: (country) => model.catalogData.developer.country === country,
    }))
  const ladders = Array.groupBy(catalogModels, (model) => {
    const parsed = parseModelId(model.modelId)
    return parsed._tag === "Catalog" ? parsed.baseId : model.modelId
  })
  const ranked = (freeBytes: Option.Option<number>) => {
    let measured = 0
    let rungs = 0
    const models = Object.entries(ladders).flatMap(([base, variants]): readonly RecommendedModel[] => {
      const ladder = Array.sort(variants, Order.mapInput(Order.number, (model: CatalogLocalModel) => model.storageBytes))
        .map((model) => rung(model, freeBytes))
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
    return { measured, rungs, models: Array.sort(models, recommendationOrder) }
  }
  const freeBytes = Option.map(inputs.hardware.modelStorage, ({ freeBytes }) => freeBytes)
  const { measured, rungs, models } = ranked(freeBytes)
  const byId = new Map(catalogModels.map((model) => [model.modelId as string, model] as const))
  // The model that would lead with unlimited disk, when it is not the one leading here because its rung lacks room.
  const leader = (ranking: readonly RecommendedModel[]) => Array.findFirst(ranking, (model) => model.meetsOnboardingFloor)
  const leading = Option.map(leader(models), ({ base }) => base)
  const roomShortfall = Option.flatMap(freeBytes, (free) =>
    leader(ranked(Option.none()).models).pipe(
      Option.filter((model) => !Option.contains(leading, model.base)),
      Option.flatMap((model) => Option.flatMap(model.chosen, (rung): Option.Option<RoomShortfall> => {
        const missingBytes = remainingDownloadBytes(byId.get(rung)!) - free
        return missingBytes > 0 ? Option.some({ base: model.base, name: model.name, rung, missingBytes }) : Option.none()
      })),
    ))
  return { complete: measured === rungs, measured, rungs, models, roomShortfall, families: inputs.families }
}

/**
 * Every view of the recommendations a client can ask for: all developers', then each developer country's among the
 * catalog's models. Recommendations are republished only when one of these changes, whatever input moved: free disk
 * shifting by a log write's few KiB changes no offer.
 */
export const recommendationViews = (inputs: Omit<RecommendationInputs, "developerCountry">): readonly RecommendationsState[] => {
  const countries = [...new Set(inputs.models.flatMap((model) => model._tag === "Catalog" ? [model.catalogData.developer.country] : []))].sort()
  return [Option.none<CountryCode>(), ...countries.map((country) => Option.some(country))]
    .map((developerCountry) => projectRecommendations({ ...inputs, developerCountry }))
}

export const sameRecommendationViews = Schema.equivalence(Schema.Array(RecommendationsSchema))

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
  const inputs = Effect.all({ local: local.state, source: sources.state, hardware: hardware.state }).pipe(
    Effect.map(({ local, source, hardware }) => ({
      models: local.models,
      catalog: source.catalogModels,
      families: projectModelFamilies(source.families),
      hardware,
    })),
  )
  yield* Stream.merge(local.changes, hardware.changes.pipe(Stream.as(undefined))).pipe(
    Stream.mapEffect(() => Effect.map(inputs, recommendationViews)),
    Stream.changesWith(sameRecommendationViews),
    Stream.runForEach(() => changes.publish({ operation: Models.getRecommendations._tag })),
    Effect.forkScoped,
  )
  return Recommendations.of({
    get: (developerCountry) => Effect.map(inputs, (current) => projectRecommendations({ ...current, developerCountry })),
  })
}))
