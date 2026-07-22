import { Context, Duration, Effect, Layer, Option, Ref, Schema, Stream } from "effect"
import { IcnClient } from "../client.js"
import type * as Generated from "../generated/schemas.js"
import { IcnHardware } from "../hardware/index.js"
import { IcnInventory } from "../inventory/index.js"
import { makeIcnObservedState, type IcnObservedState } from "../observed-state.js"
import {
  MODEL_RECIPE_REGISTRY,
  recipeSourcePageUrl,
  resolveModelRecipeArtifact,
} from "./catalog.js"
import { ModelRecipesState, type ModelRecipeRecommendation } from "./schema.js"
import type { ResolvedModelRecipe } from "./types.js"

const PREVIEW_REQUEST_CONCURRENCY = 12
const PROFILE_CONTEXTS = [200_000, 100_000] as const
const PROFILE_PARALLEL_SEQUENCES = 1
const MAX_RECOMMENDATIONS = 4
const MATERIALLY_LIGHTER_RATIO = 0.8
interface RecommendationResult {
  readonly recommendations: readonly ModelRecipeRecommendation[]
  readonly failureCount: number
}

interface ResolvedRecipes {
  readonly entries: readonly ResolvedModelRecipe[]
  readonly commits: readonly string[]
  readonly failureCount: number
}

export interface RankedRecommendationCandidate {
  readonly value: ModelRecipeRecommendation
  readonly modelId: string
  readonly modelQualityRank: number
  readonly fidelityRank: number
}

export interface IcnRecipesService extends IcnObservedState<ModelRecipesState, never> {
  readonly resolve: (configurationId: string) => Effect.Effect<Option.Option<ModelRecipeRecommendation>>
}

export class IcnRecipes extends Context.Tag("@magnitudedev/icn/IcnRecipes")<
  IcnRecipes,
  IcnRecipesService
>() {}

export interface IcnRecipesOptions {
  readonly refreshInterval?: Duration.DurationInput
}

const profileFor = (entry: ResolvedModelRecipe, contextLength: number) => ({
  id: `${entry.id}:p${PROFILE_PARALLEL_SEQUENCES}:ctx${contextLength}`,
  context_length: contextLength,
  parallel_sequences: PROFILE_PARALLEL_SEQUENCES,
})

const sourceFor = (entry: ResolvedModelRecipe): Generated.ModelPreviewSourceSchema => ({
  repository: entry.repo,
  revision: entry.revision,
  primary_gguf: entry.primaryGguf,
  additional_components: entry.additionalComponents,
})

type InspectedInventoryProperties = Extract<
  Generated.InventoryPropertiesSchema,
  { readonly type: "inspected" }
>

const recommendationFrom = (
  entry: ResolvedModelRecipe,
  preview: Generated.ModelPreviewSchema,
  assessment: Generated.ModelPreviewAssessmentSchema,
): Option.Option<ModelRecipeRecommendation> => {
  if (assessment.assessment.type !== "fits") return Option.none()
  const properties = preview.properties.type === "inspected"
    ? Option.some(preview.properties)
    : Option.none<InspectedInventoryProperties>()
  const profile = assessment.assessment.profile
  const memory = assessment.assessment.memory
  const nonNullProperty = <A>(read: (value: InspectedInventoryProperties) => Option.Option<A | null>) =>
    Option.flatMap(properties, (value) => Option.filter(read(value), (candidate): candidate is A => candidate !== null))
  const quantization = nonNullProperty((value) => value.quantization)
  const architectureName = nonNullProperty((value) => value.architecture)
  const trainingContext = nonNullProperty((value) => value.training_context_length)
  const totalParameters = nonNullProperty((value) => value.parameter_count)
  const activeParameters = nonNullProperty((value) => value.active_parameter_count)
  if (!preview.assessments.some((candidate) => candidate.profile_id === assessment.profile_id)) return Option.none()
  const matchedContext = Option.flatMap(
    Option.fromNullable(assessment.profile_id.match(/:ctx(\d+)$/)),
    (match) => Option.flatMap(Option.fromNullable(match.at(1)), (value) => Option.liftPredicate(Number(value), Number.isSafeInteger)),
  )
  const contextTokens = Option.getOrElse(Option.orElse(matchedContext, () => trainingContext), () => 1)
  const acceleration = profile.acceleration.toLowerCase()
  const fitClass = acceleration.includes("hybrid")
    ? "hybrid"
    : acceleration.includes("cpu")
      ? "cpu_or_unified"
      : "full_accelerator"
  const totalDownloadBytes = preview.components.reduce((total, component) => total + component.size_bytes, 0)

  return Option.some({
    configurationId: assessment.profile_id,
    catalogModelId: entry.id,
    artifactFingerprint: assessment.artifact_fingerprint,
    modelId: Option.none(),
    badge: "alternative",
    displayName: entry.displayName,
    family: entry.family,
    architecture: Option.exists(architectureName, (name) => name.toLowerCase().includes("moe")) ? "moe" : "dense",
    totalParametersBillions: Option.map(totalParameters, (value) => value / 1_000_000_000),
    activeParametersBillions: Option.map(activeParameters, (value) => value / 1_000_000_000),
    effectiveParametersBillions: Option.none(),
    quantization: {
      format: Option.getOrElse(quantization, () => entry.quantTag),
      quantAwareCheckpoint: entry.quantization.quantAwareCheckpoint,
      fidelityLabel: entry.quantization.fidelityLabel,
      fidelityEvidence: entry.quantization.fidelityEvidence,
      fidelitySourceUrl: entry.quantization.fidelitySourceUrl,
    },
    quantTag: Option.getOrElse(quantization, () => entry.quantTag),
    repo: preview.repository,
    revision: preview.commit,
    files: preview.components.map((component) => ({
      path: component.path,
      role: component.role,
      sizeBytes: component.size_bytes,
      sha256: component.content.type === "sha256" ? component.content.value : "",
    })),
    totalDownloadBytes,
    sourcePageUrl: recipeSourcePageUrl(entry),
    license: entry.license,
    contextTokens,
    modelMaximumContextTokens: Option.getOrElse(trainingContext, () => contextTokens),
    estimatedRuntimeBytes: memory.required_bytes,
    stableCapacityBudgetBytes: memory.available_bytes,
    fitMarginBytes: memory.headroom_bytes,
    fitClass,
    constrainedContext: assessment.assessment.recommendation === "constrained",
    explanation: `${entry.quantization.fidelityLabel}; ${profile.acceleration} placement at ${Math.round(contextTokens / 1_000)}K context.`,
  })
}

const compareConfiguration = (
  left: RankedRecommendationCandidate,
  right: RankedRecommendationCandidate,
): number => right.fidelityRank - left.fidelityRank
  || right.value.contextTokens - left.value.contextTokens
  || right.value.fitMarginBytes - left.value.fitMarginBytes

const compareProductRank = (
  left: RankedRecommendationCandidate,
  right: RankedRecommendationCandidate,
): number => right.modelQualityRank - left.modelQualityRank
  || right.fidelityRank - left.fidelityRank
  || right.value.contextTokens - left.value.contextTokens
  || right.value.fitMarginBytes - left.value.fitMarginBytes

export const selectRecommendationPortfolio = (
  candidates: readonly RankedRecommendationCandidate[],
): readonly ModelRecipeRecommendation[] => {
  const bestByModel = new Map<string, RankedRecommendationCandidate>()
  for (const candidate of candidates) {
    const current = bestByModel.get(candidate.modelId)
    if (!current || compareConfiguration(candidate, current) < 0) bestByModel.set(candidate.modelId, candidate)
  }

  const ranked = [...bestByModel.values()].sort(compareProductRank)
  const primary = ranked.shift()
  if (!primary) return []

  const selected: Array<{
    candidate: RankedRecommendationCandidate
    badge: ModelRecipeRecommendation["badge"]
  }> = [{ candidate: primary, badge: "recommended" }]
  const remaining = new Map(ranked.map((candidate) => [candidate.modelId, candidate]))
  const take = (
    badge: ModelRecipeRecommendation["badge"],
    predicate: (candidate: RankedRecommendationCandidate) => boolean,
  ): void => {
    const candidate = [...remaining.values()].find(predicate)
    if (!candidate || selected.length >= MAX_RECOMMENDATIONS) return
    remaining.delete(candidate.modelId)
    selected.push({ candidate, badge })
  }

  take("lighter", (candidate) => candidate.value.estimatedRuntimeBytes
    <= primary.value.estimatedRuntimeBytes * MATERIALLY_LIGHTER_RATIO)
  take("higher_fidelity", (candidate) => candidate.fidelityRank > primary.fidelityRank)
  take("alternative", (candidate) => candidate.value.family !== primary.value.family)

  for (const candidate of remaining.values()) {
    if (selected.length >= MAX_RECOMMENDATIONS) break
    selected.push({ candidate, badge: "alternative" })
  }
  return selected.map(({ candidate, badge }) => ({ ...candidate.value, badge }))
}

export const makeIcnRecipes = (
  options: IcnRecipesOptions = {},
): Layer.Layer<IcnRecipes, never, IcnClient | IcnHardware | IcnInventory> => Layer.scoped(
  IcnRecipes,
  Effect.gen(function* () {
    const client = yield* IcnClient
    const hardware = yield* IcnHardware
    const inventory = yield* IcnInventory
    const cache = yield* Ref.make<ReadonlyMap<string, RecommendationResult>>(new Map())

    const resolveRecipes = Effect.gen(function* () {
      const repositories = [...new Set(MODEL_RECIPE_REGISTRY.models.flatMap((model) =>
        model.artifacts.map((artifact) => artifact.repository)))]
      const results = yield* Effect.forEach(
        repositories,
        (repository) => client.huggingFace.resolveHuggingFaceRepository({
          payload: { repository, revision: "main" },
        }).pipe(
          Effect.map((snapshot) => [repository, snapshot] as const),
          Effect.either,
        ),
        { concurrency: PREVIEW_REQUEST_CONCURRENCY },
      )
      const snapshots = new Map(results.flatMap((result) => result._tag === "Right" ? [result.right] : []))
      const entries = MODEL_RECIPE_REGISTRY.models.flatMap((model) => model.artifacts.flatMap((artifact) => {
        const snapshot = snapshots.get(artifact.repository)
        if (!snapshot) return []
        const entry = resolveModelRecipeArtifact(model, artifact, {
          repository: snapshot.repository,
          commit: snapshot.commit,
          license: Option.filter(snapshot.license, (license): license is string => license !== null),
          licenseUrl: Option.filter(snapshot.license_url, (url): url is string => url !== null),
          ggufFiles: snapshot.gguf_files.map((file) => ({
            path: file.path,
            sizeBytes: Option.some(file.size_bytes),
          })),
        })
        return Option.toArray(entry)
      }))
      return {
        entries,
        commits: [...snapshots.entries()]
          .map(([repository, snapshot]) => `${repository}@${snapshot.commit}`)
          .sort(),
        failureCount: MODEL_RECIPE_REGISTRY.models
          .reduce((count, model) => count + model.artifacts.length, 0) - entries.length,
      } satisfies ResolvedRecipes
    })

    const readRecommendations = Effect.gen(function* () {
      const hardwareSnapshot = (yield* hardware.get).state
      const recipes = yield* resolveRecipes
      const key = [hardwareSnapshot.native_build, hardwareSnapshot.topology_fingerprint, ...recipes.commits].join(":")
      const cached = (yield* Ref.get(cache)).get(key)
      if (cached) return cached
      const totalStableCapacity = hardwareSnapshot.memory_domains
        .reduce((total, domain) => total + domain.stable_capacity_bytes, 0)
      const plausibleEntries = recipes.entries
        .filter((entry) => entry.publishedWeightBytes <= totalStableCapacity)
      const previewResults = yield* Effect.forEach(
        plausibleEntries,
        (entry) => {
          const contexts = PROFILE_CONTEXTS.filter((context) => entry.supportedContextTokens.includes(context))
          return client.models.previewModel({
            payload: {
              source: sourceFor(entry),
              profiles: contexts.map((context) => profileFor(entry, context)),
            },
          }).pipe(Effect.map((preview) => ({ entry, preview })), Effect.either)
        },
        { concurrency: PREVIEW_REQUEST_CONCURRENCY },
      )
      const successful = previewResults.flatMap((result) => result._tag === "Right" ? [result.right] : [])
      const candidates = successful.flatMap(({ entry, preview }) => preview.assessments.flatMap((assessment) => {
        const value = recommendationFrom(entry, preview, assessment)
        return Option.toArray(Option.map(value, (recommendation) => ({
          value: recommendation,
          modelId: entry.modelId,
          modelQualityRank: entry.modelQualityRank,
          fidelityRank: entry.quantization.fidelityRank,
        })))
      }))
      const result = {
        recommendations: selectRecommendationPortfolio(candidates),
        failureCount: recipes.failureCount + previewResults.length - successful.length,
      }
      yield* Ref.update(cache, (current) => new Map(current).set(key, result))
      return result
    })

    const read = Effect.all({
      result: readRecommendations,
      inventory: inventory.get,
    }).pipe(Effect.map(({ result, inventory }): ModelRecipesState => ({
      _tag: "Ready",
      recommendations: result.recommendations.map((recommendation) => ({
        ...recommendation,
        modelId: Option.map(
          Option.fromNullable(inventory.state.data.find((model) => {
            const contentId = Option.flatMap(
              model.content_id,
              Option.fromNullable,
            )
            return model.source.type === "hugging_face"
              && Option.exists(contentId, (value) =>
                `${model.source.repository}:${model.source.commit}:${value}`
                  === recommendation.artifactFingerprint)
          })),
          (model) => model.id,
        ),
      })),
      failureCount: result.failureCount,
    })))
    const observed = yield* makeIcnObservedState<ModelRecipesState, never>(
      { _tag: "Loading" },
      read,
      Schema.equivalence(ModelRecipesState),
    )
    yield* observed.refresh.pipe(Effect.forkScoped)
    yield* inventory.changes.pipe(
      Stream.drop(1),
      Stream.runForEach(() => observed.refresh),
      Effect.forkScoped,
    )
    yield* observed.refresh.pipe(
      Effect.delay(options.refreshInterval ?? "1 hour"),
      Effect.forever,
      Effect.forkScoped,
    )

    return IcnRecipes.of({
      ...observed,
      resolve: (configurationId) => observed.get.pipe(Effect.map(({ state }) =>
        state._tag === "Ready"
          ? Option.fromNullable(state.recommendations.find((recommendation) =>
              recommendation.configurationId === configurationId))
          : Option.none())),
    })
  }),
)
