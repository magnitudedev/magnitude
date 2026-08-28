import * as FileSystem from "@effect/platform/FileSystem"
import { Data, Effect, Schema } from "effect"
import { basename, dirname, resolve } from "node:path"
import { pathToFileURL } from "node:url"
import type { JsonRecord } from "@magnitudedev/utils/schema"
import { ChatCompletionsRequestExtensionsSchema } from "@magnitudedev/ai"
import type { ProfileName } from "./domain"

const NonEmptyString = Schema.String.pipe(Schema.minLength(1))
const PositiveInt = Schema.Int.pipe(Schema.greaterThan(0))
const Sha256 = Schema.String.pipe(Schema.pattern(/^[a-f0-9]{64}$/))

export const ModelId = NonEmptyString.pipe(Schema.brand("ModelId"))
export type ModelId = typeof ModelId.Type
export const ExperimentId = NonEmptyString.pipe(Schema.brand("ExperimentId"))
export type ExperimentId = typeof ExperimentId.Type
export const VariantId = NonEmptyString.pipe(Schema.brand("VariantId"))
export type VariantId = typeof VariantId.Type

export interface GgufArtifactDefinition {
  readonly kind: "gguf"
  readonly modelId: ModelId
  readonly modelSource: { readonly repository: string; readonly revision: string }
  readonly modelContextLimit: number
  readonly repository: string
  readonly revision: string
  readonly file: string
  readonly sizeBytes: number
  readonly sha256: string
  readonly quantization: { readonly family: "gguf"; readonly scheme: string }
}

export interface MlxArtifactDefinition {
  readonly kind: "mlx"
  readonly modelId: ModelId
  readonly modelSource: { readonly repository: string; readonly revision: string }
  readonly modelContextLimit: number
  readonly repository: string
  readonly revision: string
  readonly manifest: string
  readonly quantization:
    | { readonly family: "mlx-affine"; readonly bits: number; readonly groupSize: number }
    | { readonly family: "mlx-unquantized"; readonly dtype: "bfloat16" | "float16" }
}

export type ModelArtifactDefinition = GgufArtifactDefinition | MlxArtifactDefinition
export type ModelArtifactInput =
  | Omit<GgufArtifactDefinition, "modelId" | "modelSource" | "modelContextLimit">
  | Omit<MlxArtifactDefinition, "modelId" | "modelSource" | "modelContextLimit">

export interface ModelDefinition<Artifacts extends Readonly<Record<string, ModelArtifactInput>>> {
  readonly id: ModelId
  readonly source: { readonly repository: string; readonly revision: string }
  readonly contextLimit: number
  readonly artifacts: {
    readonly [K in keyof Artifacts]: Artifacts[K] & {
      readonly modelId: ModelId
      readonly modelSource: { readonly repository: string; readonly revision: string }
      readonly modelContextLimit: number
    }
  }
}

export interface LlamaCppEngineDefinition {
  readonly kind: "llama.cpp"
  readonly executable: "managed" | string
  readonly flashAttention: boolean
  readonly continuousBatching: boolean
  readonly kvCache: { readonly quantization: "none" }
  readonly speculativeDecoding:
    | { readonly kind: "none" }
    | { readonly kind: "mtp"; readonly draftArtifact: ModelArtifactDefinition; readonly maxDraftTokens: number }
}

export interface MlxLmEngineDefinition {
  readonly kind: "mlx-lm"
  readonly pythonProject: string
  readonly prefillStepSize: number
  readonly promptCacheEntries: number
  readonly kvCache: { readonly quantization: "none" }
  readonly speculativeDecoding: { readonly kind: "none" }
}

export interface MlxVlmEngineDefinition {
  readonly kind: "mlx-vlm"
  readonly pythonProject: string
  readonly prefillStepSize: number
  readonly kvCache: { readonly quantization: "none" }
  readonly speculativeDecoding: {
    readonly kind: "mtp"
    readonly draftArtifact: ModelArtifactDefinition
    readonly maxDraftTokens: number
  }
}

export interface IcnEngineDefinition {
  readonly kind: "icn"
  readonly executable: "managed" | string
}

export interface ExistingEndpointEngineDefinition {
  readonly kind: "existing-endpoint"
  readonly endpoint: string
  readonly authentication: { readonly kind: "none" } | { readonly kind: "bearer-env"; readonly variable: string }
  readonly requestBody: JsonRecord
}

export type OmlxSpeculation =
  | { readonly kind: "none" }
  | { readonly kind: "mtp"; readonly maxDraftTokens: number }
  | { readonly kind: "dspark"; readonly maxDraftTokens: number }
  | { readonly kind: "dflash"; readonly draftArtifact: MlxArtifactDefinition; readonly blockSize: number }

export interface OmlxEngineDefinition {
  readonly kind: "omlx"
  readonly pythonProject: string
  readonly cache: { readonly kind: "disabled" }
  readonly memoryGuard: { readonly kind: "off" }
  readonly speculativeDecoding: OmlxSpeculation
}

export type EngineDefinition = LlamaCppEngineDefinition | MlxLmEngineDefinition | MlxVlmEngineDefinition | OmlxEngineDefinition | IcnEngineDefinition | ExistingEndpointEngineDefinition

export type ComparisonProtocol =
  | { readonly kind: "fixed-speculative-policy" }
  | { readonly kind: "speculative-decoding" }

export interface ContextSweepDefinition {
  readonly kind: "context-sweep"
  readonly checkpoints: readonly number[]
  readonly charactersPerToken: number
  readonly samplesPerCheckpoint: number
}

export interface ExperimentDefinition {
  readonly id: ExperimentId
  readonly title: string
  readonly comparisonProtocol: ComparisonProtocol
  readonly suite:
    | { readonly kind: "agent-core"; readonly profile: ProfileName }
    | ContextSweepDefinition
  readonly requestPolicy: {
    readonly contextTokensPerSequence: number
    readonly parallelSequences: number
    readonly maxOutputTokens: number
    readonly requestTimeoutMs: number
    readonly temperature: number
    readonly topP: number
    readonly seed: number
    readonly enableThinking: false
  }
  readonly variants: readonly {
    readonly id: VariantId
    readonly artifact: ModelArtifactDefinition
    readonly engine: EngineDefinition
  }[]
  readonly execution: { readonly variantOrder: "balanced" | "declared"; readonly blocks: number }
}

export type ExperimentInput = Omit<ExperimentDefinition, "id" | "variants" | "requestPolicy" | "comparisonProtocol"> & {
  readonly id: string
  readonly requestPolicy: Omit<ExperimentDefinition["requestPolicy"], "requestTimeoutMs"> & {
    readonly requestTimeoutMs?: number
  }
  readonly variants: readonly (Omit<ExperimentDefinition["variants"][number], "id"> & { readonly id: string })[]
}

const Artifact = Schema.Union(
  Schema.Struct({
    kind: Schema.Literal("gguf"), modelId: ModelId,
    modelSource: Schema.Struct({ repository: NonEmptyString, revision: NonEmptyString }),
    modelContextLimit: PositiveInt,
    repository: NonEmptyString,
    revision: NonEmptyString, file: NonEmptyString, sizeBytes: PositiveInt, sha256: Sha256,
    quantization: Schema.Struct({ family: Schema.Literal("gguf"), scheme: NonEmptyString }),
  }),
  Schema.Struct({
    kind: Schema.Literal("mlx"), modelId: ModelId,
    modelSource: Schema.Struct({ repository: NonEmptyString, revision: NonEmptyString }),
    modelContextLimit: PositiveInt,
    repository: NonEmptyString,
    revision: NonEmptyString, manifest: NonEmptyString,
    quantization: Schema.Union(
      Schema.Struct({ family: Schema.Literal("mlx-affine"), bits: PositiveInt, groupSize: PositiveInt }),
      Schema.Struct({ family: Schema.Literal("mlx-unquantized"), dtype: Schema.Literal("bfloat16", "float16") }),
    ),
  }),
)
const SpeculativeNone = Schema.Struct({ kind: Schema.Literal("none") })
const SpeculativeMtp = Schema.Struct({
  kind: Schema.Literal("mtp"),
  draftArtifact: Artifact,
  maxDraftTokens: PositiveInt,
})
const OmlxSpeculationSchema = Schema.Union(
  SpeculativeNone,
  Schema.Struct({ kind: Schema.Literal("mtp"), maxDraftTokens: PositiveInt }),
  Schema.Struct({ kind: Schema.Literal("dspark"), maxDraftTokens: PositiveInt }),
  Schema.Struct({
    kind: Schema.Literal("dflash"),
    draftArtifact: Artifact.pipe(Schema.filter((artifact): artifact is MlxArtifactDefinition => artifact.kind === "mlx", {
      message: () => "oMLX DFlash requires an MLX draft artifact",
    })),
    blockSize: PositiveInt,
  }),
)
const Engine = Schema.Union(
  Schema.Struct({
    kind: Schema.Literal("llama.cpp"), executable: NonEmptyString,
    flashAttention: Schema.Boolean, continuousBatching: Schema.Boolean,
    kvCache: Schema.Struct({ quantization: Schema.Literal("none") }),
    speculativeDecoding: Schema.Union(SpeculativeNone, SpeculativeMtp),
  }),
  Schema.Struct({
    kind: Schema.Literal("mlx-lm"), pythonProject: NonEmptyString,
    prefillStepSize: PositiveInt, promptCacheEntries: PositiveInt,
    kvCache: Schema.Struct({ quantization: Schema.Literal("none") }),
    speculativeDecoding: SpeculativeNone,
  }),
  Schema.Struct({
    kind: Schema.Literal("mlx-vlm"), pythonProject: NonEmptyString,
    prefillStepSize: PositiveInt,
    kvCache: Schema.Struct({ quantization: Schema.Literal("none") }),
    speculativeDecoding: SpeculativeMtp,
  }),
  Schema.Struct({
    kind: Schema.Literal("omlx"), pythonProject: NonEmptyString,
    cache: Schema.Struct({ kind: Schema.Literal("disabled") }),
    memoryGuard: Schema.Struct({ kind: Schema.Literal("off") }),
    speculativeDecoding: OmlxSpeculationSchema,
  }),
  Schema.Struct({
    kind: Schema.Literal("icn"), executable: NonEmptyString,
  }),
  Schema.Struct({
    kind: Schema.Literal("existing-endpoint"), endpoint: NonEmptyString,
    authentication: Schema.Union(
      Schema.Struct({ kind: Schema.Literal("none") }),
      Schema.Struct({ kind: Schema.Literal("bearer-env"), variable: NonEmptyString }),
    ),
    requestBody: ChatCompletionsRequestExtensionsSchema,
  }),
)
export const ExperimentDefinitionSchema = Schema.Struct({
  id: ExperimentId,
  title: NonEmptyString,
  comparisonProtocol: Schema.Union(
    Schema.Struct({ kind: Schema.Literal("fixed-speculative-policy") }),
    Schema.Struct({ kind: Schema.Literal("speculative-decoding") }),
  ),
  suite: Schema.Union(
    Schema.Struct({ kind: Schema.Literal("agent-core"), profile: Schema.Literal("smoke", "standard", "full") }),
    Schema.Struct({
      kind: Schema.Literal("context-sweep"),
      checkpoints: Schema.NonEmptyArray(PositiveInt),
      charactersPerToken: Schema.Number.pipe(Schema.greaterThan(0)),
      samplesPerCheckpoint: PositiveInt,
    }),
  ),
  requestPolicy: Schema.Struct({
    contextTokensPerSequence: PositiveInt, parallelSequences: PositiveInt, maxOutputTokens: PositiveInt,
    requestTimeoutMs: Schema.optionalWith(PositiveInt, { default: () => 300_000 }),
    temperature: Schema.Number.pipe(Schema.nonNegative()),
    topP: Schema.Number.pipe(Schema.greaterThan(0), Schema.lessThanOrEqualTo(1)),
    seed: Schema.Int, enableThinking: Schema.Literal(false),
  }),
  variants: Schema.NonEmptyArray(Schema.Struct({ id: VariantId, artifact: Artifact, engine: Engine })),
  execution: Schema.Struct({ variantOrder: Schema.Literal("balanced", "declared"), blocks: PositiveInt }),
})

export class ExperimentError extends Data.TaggedError("ExperimentError")<{
  readonly path: string
  readonly operation: "discover" | "load" | "validate"
  readonly message: string
}> {}

function assertSerializable(value: unknown, seen = new Set<object>(), path = "experiment"): void {
  if (value === null || typeof value === "string" || typeof value === "boolean") return
  if (typeof value === "number" && Number.isFinite(value)) return
  if (typeof value !== "object") throw new Error(`${path} contains non-serializable ${typeof value}`)
  if (seen.has(value)) throw new Error(`${path} contains a cycle`)
  seen.add(value)
  if (Array.isArray(value)) value.forEach((item, index) => assertSerializable(item, seen, `${path}[${index}]`))
  else {
    const prototype = Object.getPrototypeOf(value)
    if (prototype !== Object.prototype && prototype !== null) throw new Error(`${path} contains a non-plain object`)
    for (const [key, item] of Object.entries(value)) assertSerializable(item, seen, `${path}.${key}`)
  }
  seen.delete(value)
}

export function defineModel<const Artifacts extends Readonly<Record<string, ModelArtifactInput>>>(definition: {
  readonly id: string
  readonly source: { readonly repository: string; readonly revision: string }
  readonly contextLimit: number
  readonly artifacts: Artifacts
}): ModelDefinition<Artifacts> {
  if (!Number.isSafeInteger(definition.contextLimit) || definition.contextLimit <= 0) {
    throw new Error("model context limit must be a positive integer")
  }
  const id = definition.id as ModelId
  return {
    id,
    source: definition.source,
    contextLimit: definition.contextLimit,
    artifacts: Object.fromEntries(Object.entries(definition.artifacts).map(([key, artifact]) => [key, {
      ...artifact,
      modelId: id,
      modelSource: definition.source,
      modelContextLimit: definition.contextLimit,
    }])) as unknown as ModelDefinition<Artifacts>["artifacts"],
  }
}

export const agentCore = (options: { readonly profile: ProfileName }) => ({ kind: "agent-core" as const, ...options })
export const contextSweep = (options: Omit<ContextSweepDefinition, "kind">): ContextSweepDefinition => ({ kind: "context-sweep", ...options })
export const llamaCpp = (options: Omit<LlamaCppEngineDefinition, "kind">): LlamaCppEngineDefinition => ({ kind: "llama.cpp", ...options })
export const mlxLm = (options: Omit<MlxLmEngineDefinition, "kind">): MlxLmEngineDefinition => ({ kind: "mlx-lm", ...options })
export const mlxVlm = (options: Omit<MlxVlmEngineDefinition, "kind">): MlxVlmEngineDefinition => ({ kind: "mlx-vlm", ...options })
export const omlx = (options: Omit<OmlxEngineDefinition, "kind">): OmlxEngineDefinition => ({ kind: "omlx", ...options })
export const icn = (options: Omit<IcnEngineDefinition, "kind">): IcnEngineDefinition => ({ kind: "icn", ...options })
export const existingEndpoint = (options: Omit<ExistingEndpointEngineDefinition, "kind">): ExistingEndpointEngineDefinition => ({ kind: "existing-endpoint", ...options })

function validateExperimentDefinition(definition: unknown): ExperimentDefinition {
  assertSerializable(definition)
  const decoded = Schema.decodeUnknownSync(ExperimentDefinitionSchema)(definition) as ExperimentDefinition
  const ids = new Set(decoded.variants.map(({ id }) => id))
  if (ids.size !== decoded.variants.length) throw new Error("variant ids must be unique")
  const models = new Set(decoded.variants.map(({ artifact }) => artifact.modelId))
  if (models.size !== 1) throw new Error("the first benchmark implementation requires one logical model")
  const modelContexts = new Set(decoded.variants.map(({ artifact }) => artifact.modelContextLimit))
  if (modelContexts.size !== 1) throw new Error("comparable artifacts must declare the same logical model context limit")
  const modelContextLimit = decoded.variants[0]!.artifact.modelContextLimit
  if (decoded.requestPolicy.contextTokensPerSequence > modelContextLimit) {
    throw new Error("configured context per sequence exceeds the logical model context limit")
  }
  if (decoded.suite.kind === "context-sweep") {
    const checkpoints = decoded.suite.checkpoints
    if (new Set(checkpoints).size !== checkpoints.length
      || checkpoints.some((checkpoint, index) => index > 0 && checkpoint <= checkpoints[index - 1]!)) {
      throw new Error("context-sweep checkpoints must be unique and strictly increasing")
    }
    const maximum = checkpoints[checkpoints.length - 1]!
    if (maximum + decoded.requestPolicy.maxOutputTokens > decoded.requestPolicy.contextTokensPerSequence) {
      throw new Error("context-sweep maximum plus output allowance exceeds the configured context window")
    }
  }
  const omlxVariants = decoded.variants.filter(({ engine }) => engine.kind === "omlx")
  if (omlxVariants.length > 0 && omlxVariants.length !== decoded.variants.length) {
    throw new Error("managed oMLX variants cannot be mixed with another serving runtime")
  }
  if (omlxVariants.length > 0
    && new Set(omlxVariants.map(({ engine }) => (engine as OmlxEngineDefinition).pythonProject)).size !== 1) {
    throw new Error("managed oMLX variants must use one frozen Python project")
  }
  for (const variant of decoded.variants) {
    if (variant.engine.kind === "icn" && variant.artifact.kind !== "gguf") {
      throw new Error(`ICN variant ${variant.id} requires a GGUF artifact`)
    }
    if (variant.engine.kind === "llama.cpp" && variant.engine.speculativeDecoding.kind === "mtp" && variant.engine.speculativeDecoding.draftArtifact.kind !== "gguf") {
      throw new Error(`llama.cpp MTP variant ${variant.id} requires a GGUF drafter`)
    }
    if (variant.engine.kind === "mlx-vlm" && variant.engine.speculativeDecoding.draftArtifact.kind !== "mlx") {
      throw new Error(`MLX-VLM MTP variant ${variant.id} requires an MLX drafter`)
    }
    if ((variant.engine.kind === "llama.cpp" || variant.engine.kind === "mlx-vlm")
      && variant.engine.speculativeDecoding.kind === "mtp"
      && variant.engine.speculativeDecoding.draftArtifact.modelId !== variant.artifact.modelId) {
      throw new Error(`MTP variant ${variant.id} requires a drafter for the same logical model`)
    }
    if (variant.engine.kind === "omlx" && variant.artifact.kind !== "mlx") {
      throw new Error(`oMLX variant ${variant.id} requires an MLX target artifact`)
    }
    if (variant.engine.kind === "omlx" && variant.engine.speculativeDecoding.kind === "dflash"
      && variant.engine.speculativeDecoding.draftArtifact.modelId !== variant.artifact.modelId) {
      throw new Error(`oMLX DFlash variant ${variant.id} requires a draft for the same logical model`)
    }
    if (variant.engine.kind === "omlx" && variant.engine.speculativeDecoding.kind === "dflash"
      && decoded.requestPolicy.parallelSequences !== 1) {
      throw new Error(`oMLX DFlash variant ${variant.id} requires parallelSequences=1`)
    }
  }
  if (decoded.comparisonProtocol.kind === "fixed-speculative-policy") {
    const acceleration = new Set(decoded.variants.map(({ engine }) =>
      engine.kind === "llama.cpp" || engine.kind === "mlx-lm" || engine.kind === "mlx-vlm" || engine.kind === "omlx"
        ? engine.speculativeDecoding.kind
        : "unspecified"))
    if (acceleration.size !== 1) throw new Error("fixed-speculative-policy comparisons require the same speculative-decoding mode")
    const draftLimits = new Set(decoded.variants.flatMap(({ engine }) =>
      ((engine.kind === "llama.cpp" || engine.kind === "mlx-vlm") && engine.speculativeDecoding.kind === "mtp")
        || (engine.kind === "omlx" && (engine.speculativeDecoding.kind === "mtp" || engine.speculativeDecoding.kind === "dspark"))
        ? [engine.speculativeDecoding.maxDraftTokens]
        : []))
    if (draftLimits.size > 1) throw new Error("fixed-speculative-policy comparisons require the same maximum draft-token count")
  } else {
    if (decoded.variants.length < 2 || decoded.variants.some(({ engine }) => engine.kind !== "omlx")) {
      throw new Error("speculative-decoding comparisons require at least two oMLX variants")
    }
    const omlxEngines = decoded.variants.map(({ engine }) => engine as OmlxEngineDefinition)
    const modes = new Set(omlxEngines.map(({ speculativeDecoding }) => speculativeDecoding.kind))
    if (!modes.has("none") || modes.size < 2) {
      throw new Error("speculative-decoding comparisons require a baseline and a speculative method")
    }
    const targetIdentity = digestComparableArtifact(decoded.variants[0]!.artifact)
    if (decoded.variants.some(({ artifact }) => digestComparableArtifact(artifact) !== targetIdentity)) {
      throw new Error("speculative-decoding comparisons require the exact same target artifact")
    }
    const engineIdentity = JSON.stringify({
      pythonProject: omlxEngines[0]!.pythonProject,
      cache: omlxEngines[0]!.cache,
      memoryGuard: omlxEngines[0]!.memoryGuard,
    })
    if (omlxEngines.some((engine) => JSON.stringify({ pythonProject: engine.pythonProject, cache: engine.cache, memoryGuard: engine.memoryGuard }) !== engineIdentity)) {
      throw new Error("speculative-decoding comparisons may differ only by speculative policy")
    }
    if (decoded.execution.variantOrder !== "balanced" || decoded.execution.blocks % decoded.variants.length !== 0) {
      throw new Error("speculative-decoding comparisons require balanced blocks divisible by the variant count")
    }
  }
  return Object.freeze(decoded)
}

export const defineExperiment = (definition: ExperimentInput): ExperimentDefinition =>
  validateExperimentDefinition({ ...definition, comparisonProtocol: { kind: "fixed-speculative-policy" } })

export const defineSpeculativeDecodingComparison = (definition: ExperimentInput): ExperimentDefinition =>
  validateExperimentDefinition({ ...definition, comparisonProtocol: { kind: "speculative-decoding" } })

function digestComparableArtifact(artifact: ModelArtifactDefinition): string {
  return JSON.stringify(artifact)
}

export const loadExperiment = (path: string): Effect.Effect<ExperimentDefinition, ExperimentError, FileSystem.FileSystem> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const absolute = resolve(path)
    if (!(yield* fs.exists(absolute).pipe(Effect.mapError((error) => new ExperimentError({ path: absolute, operation: "load", message: String(error) }))))) {
      return yield* new ExperimentError({ path: absolute, operation: "load", message: "experiment file does not exist" })
    }
    const module = yield* Effect.tryPromise({
      try: () => import(`${pathToFileURL(absolute).href}?loaded=${Date.now()}`),
      catch: (error) => new ExperimentError({ path: absolute, operation: "load", message: error instanceof Error ? error.message : String(error) }),
    })
    return yield* Effect.try({
      try: () => validateExperimentDefinition(module.default),
      catch: (error) => new ExperimentError({ path: absolute, operation: "validate", message: error instanceof Error ? error.message : String(error) }),
    })
  })

export interface DiscoveredExperiment {
  readonly path: string
  readonly fileName: string
}

export const discoverExperiments = (
  root = resolve("packages/inference-benchmark/experiments"),
): Effect.Effect<readonly DiscoveredExperiment[], ExperimentError, FileSystem.FileSystem> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const entries = yield* fs.readDirectory(root).pipe(Effect.mapError((error) => new ExperimentError({
      path: root, operation: "discover", message: error instanceof Error ? error.message : String(error),
    })))
    return entries.filter((entry) => entry.endsWith(".experiment.ts")).sort().map((entry) => ({
      path: resolve(root, entry), fileName: basename(entry),
    }))
  })

export function resolveExperimentPaths(experiment: ExperimentDefinition, experimentPath: string): ExperimentDefinition {
  const root = dirname(resolve(experimentPath))
  const resolveArtifact = (artifact: ModelArtifactDefinition): ModelArtifactDefinition => artifact.kind === "mlx"
    ? { ...artifact, manifest: resolve(root, artifact.manifest) }
    : artifact
  return {
    ...experiment,
    variants: experiment.variants.map((variant) => ({
      ...variant,
      artifact: resolveArtifact(variant.artifact),
      engine: variant.engine.kind === "mlx-lm"
        ? { ...variant.engine, pythonProject: resolve(root, variant.engine.pythonProject) }
        : variant.engine.kind === "mlx-vlm"
          ? {
              ...variant.engine,
              pythonProject: resolve(root, variant.engine.pythonProject),
              speculativeDecoding: {
                ...variant.engine.speculativeDecoding,
                draftArtifact: resolveArtifact(variant.engine.speculativeDecoding.draftArtifact),
              },
            }
        : variant.engine.kind === "omlx"
          ? {
              ...variant.engine,
              pythonProject: resolve(root, variant.engine.pythonProject),
              speculativeDecoding: variant.engine.speculativeDecoding.kind === "dflash"
                ? { ...variant.engine.speculativeDecoding, draftArtifact: resolveArtifact(variant.engine.speculativeDecoding.draftArtifact) as MlxArtifactDefinition }
                : variant.engine.speculativeDecoding,
            }
        : variant.engine.kind === "llama.cpp" && variant.engine.speculativeDecoding.kind === "mtp"
          ? { ...variant.engine, speculativeDecoding: { ...variant.engine.speculativeDecoding, draftArtifact: resolveArtifact(variant.engine.speculativeDecoding.draftArtifact) } }
          : variant.engine,
    })),
  }
}

export function resolveExecutionOrder(experiment: ExperimentDefinition): readonly (readonly VariantId[])[] {
  const declared = experiment.variants.map(({ id }) => id)
  if (experiment.comparisonProtocol.kind === "speculative-decoding" && experiment.execution.variantOrder === "balanced") {
    return Array.from({ length: experiment.execution.blocks }, (_, block) => [
      ...declared.slice(block % declared.length),
      ...declared.slice(0, block % declared.length),
    ])
  }
  return Array.from({ length: experiment.execution.blocks }, (_, block) =>
    experiment.execution.variantOrder === "balanced" && block % 2 === 1
      ? [...declared].reverse()
      : declared)
}
