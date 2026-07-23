import { Context, Duration, Effect, Layer, Option, Ref, Schema, Stream } from "effect"
import { ReasoningEffortSchema, type ReasoningEffort } from "@magnitudedev/ai"
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
import {
  RECOMMENDATION_POLICY_VERSION,
  selectRecommendationPortfolio,
  type RecommendationCandidate,
} from "./recommendation-policy.js"
import {
  ModelArtifactFingerprintSchema,
  ModelRecipeCatalogModelIdSchema,
  ModelRecipeConfigurationIdSchema,
  NativeIcnModelIdSchema,
  type ModelRecipeConfigurationId,
} from "../provider/model-identity.js"

const PREVIEW_REQUEST_CONCURRENCY = 12
const PROFILE_CONTEXTS = [200_000, 100_000] as const
const PROFILE_PARALLEL_SEQUENCES = 1
interface RecommendationResult {
  readonly recommendations: readonly ModelRecipeRecommendation[]
  readonly failureCount: number
}

interface ResolvedRecipes {
  readonly entries: readonly ResolvedModelRecipe[]
  readonly commits: readonly string[]
  readonly failureCount: number
}

export interface IcnRecipesService extends IcnObservedState<ModelRecipesState, never> {
  readonly resolve: (configurationId: ModelRecipeConfigurationId) => Effect.Effect<Option.Option<ModelRecipeRecommendation>>
}

export const recommendationCacheKey = (
  policyVersion: string,
  nativeBuild: string,
  topologyFingerprint: string,
  commits: readonly string[],
): string => [policyVersion, nativeBuild, topologyFingerprint, ...commits].join(":")

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

export const exactRequestedProfileContext = (
  profileId: string,
  actualContext: number,
): Option.Option<100_000 | 200_000> => Option.flatMap(
  Option.fromNullable(profileId.match(/:ctx(\d+)$/)?.at(1)),
  (value) => {
    const requested = Number(value)
    return requested === actualContext && (requested === 100_000 || requested === 200_000)
      ? Option.some(requested)
      : Option.none()
  },
)

export const exactGenerationEstimate = (
  performance: Generated.GenerationPerformanceAssessmentSchema,
  context: number,
) => performance.status === "estimated"
  ? Option.map(
    Option.fromNullable(performance.points.find(({ context_tokens }) => context_tokens === context)),
    (point) => ({
      contextTokens: point.context_tokens,
      lowerTokensPerSecond: point.lower_tokens_per_second,
      expectedTokensPerSecond: point.expected_tokens_per_second,
      upperTokensPerSecond: point.upper_tokens_per_second,
      confidence: performance.confidence,
      method: performance.method,
    }),
  )
  : Option.none()

const recommendationFrom = (
  entry: ResolvedModelRecipe,
  preview: Generated.ModelPreviewSchema,
  assessment: Generated.ModelPreviewAssessmentSchema,
): Option.Option<ModelRecipeRecommendation> => {
  if (assessment.assessment.type !== "fits") return Option.none()
  const properties = preview.properties.type === "inspected"
    ? Option.some(preview.properties)
    : Option.none<InspectedInventoryProperties>()
  if (Option.isNone(properties)) return Option.none()
  const profile = assessment.assessment.profile
  const memory = assessment.assessment.memory
  const nonNullProperty = <A>(read: (value: InspectedInventoryProperties) => Option.Option<A | null>) =>
    Option.flatMap(properties, (value) => Option.filter(read(value), (candidate): candidate is A => candidate !== null))
  const quantization = nonNullProperty((value) => value.quantization)
  const architectureName = nonNullProperty((value) => value.architecture)
  const totalParameters = nonNullProperty((value) => value.parameter_count)
  const activeParameters = nonNullProperty((value) => value.active_parameter_count)
  if (!preview.assessments.some((candidate) => candidate.profile_id === assessment.profile_id)) return Option.none()
  const matchedContext = exactRequestedProfileContext(
    assessment.profile_id,
    profile.context_length,
  )
  if (Option.isNone(matchedContext)) return Option.none()
  const contextWindow = matchedContext.value
  const acceleration = profile.acceleration.toLowerCase()
  const fitClass = acceleration.includes("hybrid")
    ? "hybrid"
    : acceleration.includes("cpu")
      ? "cpu_or_unified"
      : "full_accelerator"
  const totalDownloadBytes = preview.components.reduce((total, component) => total + component.size_bytes, 0)
  const reasoning = properties.value.reasoning
  const reasoningEfforts: readonly ReasoningEffort[] = (reasoning.type === "unsupported"
    ? []
    : reasoning.control.type === "effort" || reasoning.control.type === "effort_and_budget"
      ? [...new Set(reasoning.control.levels)]
      : reasoning.control.type === "toggle"
        ? ["none", "medium"]
        : ["medium"]).map((effort) => ReasoningEffortSchema.make(effort))
  const requestedDefault = reasoning.type === "unsupported"
    ? Option.none<ReasoningEffort>()
    : reasoning.control.type === "effort"
      ? Option.map(Option.filter(reasoning.control.default, (value): value is string => value !== null), ReasoningEffortSchema.make)
      : reasoning.control.type === "effort_and_budget"
        ? Option.map(Option.filter(reasoning.control.default_effort, (value): value is string => value !== null), ReasoningEffortSchema.make)
        : reasoning.control.type === "toggle"
          ? Option.some(ReasoningEffortSchema.make(reasoning.control.default ? "medium" : "none"))
          : Option.some(ReasoningEffortSchema.make("medium"))
  const defaultReasoningEffort = Option.orElse(
    Option.filter(requestedDefault, (effort) => reasoningEfforts.includes(effort)),
    () => Option.fromNullable(reasoningEfforts[0]),
  )

  return Option.some({
    configurationId: ModelRecipeConfigurationIdSchema.make(assessment.profile_id),
    catalogModelId: ModelRecipeCatalogModelIdSchema.make(entry.id),
    artifactFingerprint: ModelArtifactFingerprintSchema.make(assessment.artifact_fingerprint),
    modelId: Option.none(),
    intent: "balanced",
    displayName: entry.displayName,
    family: entry.family,
    architecture: Option.exists(architectureName, (name) => name.toLowerCase().includes("moe")) ? "moe" : "dense",
    capabilities: {
      vision: properties.value.modalities.includes("vision"),
      tools: properties.value.tools.type === "supported",
      structuredOutput: properties.value.structured_output.type === "supported",
      reasoningEfforts,
      defaultReasoningEffort,
    },
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
    contextWindow,
    estimatedRuntimeBytes: memory.required_bytes,
    stableCapacityBudgetBytes: memory.available_bytes,
    fitMarginBytes: memory.headroom_bytes,
    fitClass,
    constrainedContext: assessment.assessment.recommendation === "constrained",
    estimatedGeneration: exactGenerationEstimate(assessment.performance, contextWindow),
    explanation: `${entry.quantization.fidelityLabel}; ${profile.acceleration} placement at ${Math.round(contextWindow / 1_000)}K context.`,
  })
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
      const key = recommendationCacheKey(
        RECOMMENDATION_POLICY_VERSION,
        hardwareSnapshot.native_build,
        hardwareSnapshot.topology_fingerprint,
        recipes.commits,
      )
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
      const candidates: readonly RecommendationCandidate[] = successful.flatMap(({ entry, preview }) => preview.assessments.flatMap((assessment) => {
        const value = recommendationFrom(entry, preview, assessment)
        return Option.toArray(Option.map(value, (recommendation) => ({
          value: recommendation,
          artifactId: entry.id,
          checkpointId: entry.modelId,
          capability: entry.capability,
          fidelityRank: entry.quantization.fidelityRank,
        })))
      }))
      const recommendations = selectRecommendationPortfolio(candidates)
      const result = {
        recommendations,
        failureCount: recipes.failureCount + previewResults.length - successful.length,
      }
      yield* Effect.logDebug("Selected local model recommendation portfolio").pipe(
        Effect.annotateLogs({
          policy: RECOMMENDATION_POLICY_VERSION,
          candidates: candidates.map((candidate) => [
            candidate.value.configurationId,
            candidate.checkpointId,
            candidate.capability?.score ?? "unmeasured",
            candidate.fidelityRank,
            candidate.value.contextWindow,
            Option.getOrUndefined(candidate.value.estimatedGeneration)?.expectedTokensPerSecond
              ?? "unavailable",
            candidate.value.estimatedRuntimeBytes,
            candidate.value.totalDownloadBytes,
          ].join(":")),
          selected: recommendations.map(({ configurationId, intent }) =>
            `${configurationId}:${intent}`),
        }),
      )
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
          (model) => NativeIcnModelIdSchema.make(model.id),
        ),
      })),
      failureCount: result.failureCount,
    })))
    const observed = yield* makeIcnObservedState<ModelRecipesState, never>(
      { _tag: "Loading" },
      read,
      Schema.equivalence(ModelRecipesState),
    )
    const initialInventorySnapshot = yield* inventory.get
    yield* observed.refresh.pipe(Effect.forkScoped)
    yield* inventory.changes.pipe(
      Stream.dropWhile((snapshot) => snapshot.revision <= initialInventorySnapshot.revision),
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
