import { Effect, Option } from "effect"
import {
  AVAILABLE_PROVIDER_MODEL,
  ModelCatalogError,
  ProviderIdSchema,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
  ReasoningProperty,
  VisionProperty,
  type ModelCatalog,
} from "@magnitudedev/ai"
import { classifyModelFamily } from "../family-registry"
import type { MiniMaxModelInfo } from "./contract"

export const MINIMAX_PROVIDER_ID = ProviderIdSchema.make("minimax")

const model = (config: {
  readonly id: string
  readonly contextWindow: number
  readonly maxOutputTokens: number
  readonly vision: boolean
  readonly reasoningEfforts: readonly string[]
  readonly defaultReasoningEffort: string
  readonly pricing: {
    readonly input: number
    readonly output: number
    readonly cachedInput: number
  }
}): MiniMaxModelInfo => ({
  providerId: MINIMAX_PROVIDER_ID,
  providerModelId: ProviderModelIdSchema.make(config.id),
  modelFamilyId: Option.getOrUndefined(classifyModelFamily(config.id)),
  displayName: config.id,
  contextWindow: config.contextWindow,
  maxOutputTokens: config.maxOutputTokens,
  defaultReasoningEffort: ReasoningEffortSchema.make(config.defaultReasoningEffort),
  properties: {
    vision: new VisionProperty.states.Resolved({ value: config.vision }),
    reasoning: new ReasoningProperty.states.Resolved({
      value: config.reasoningEfforts.map((effort) => ReasoningEffortSchema.make(effort)),
    }),
  },
  servingCapabilities: { tools: true, structuredOutput: false },
  availability: AVAILABLE_PROVIDER_MODEL,
  pricing: Option.some({
    input: config.pricing.input,
    output: config.pricing.output,
    cached_input: config.pricing.cachedInput,
  }),
})

export const MINIMAX_MODELS: readonly MiniMaxModelInfo[] = [
  model({
    id: "MiniMax-M3",
    contextWindow: 1_000_000,
    maxOutputTokens: 524_288,
    vision: true,
    reasoningEfforts: ["adaptive", "disabled"],
    defaultReasoningEffort: "adaptive",
    pricing: { input: 0.6, output: 2.4, cachedInput: 0.12 },
  }),
  model({
    id: "MiniMax-M2.7",
    contextWindow: 204_800,
    maxOutputTokens: 204_800,
    vision: false,
    reasoningEfforts: ["always_on"],
    defaultReasoningEffort: "always_on",
    pricing: { input: 0.3, output: 1.2, cachedInput: 0.06 },
  }),
]

export const createMiniMaxCatalog = (): ModelCatalog<MiniMaxModelInfo> => ({
  list: Effect.succeed(MINIMAX_MODELS),
  refresh: Effect.succeed(MINIMAX_MODELS),
  get: (providerId, providerModelId) => {
    const entry = MINIMAX_MODELS.find((candidate) =>
      providerId === MINIMAX_PROVIDER_ID && candidate.providerModelId === providerModelId)
    return entry === undefined
      ? Effect.fail(new ModelCatalogError({ message: `Unknown MiniMax model: ${providerModelId}` }))
      : Effect.succeed(entry)
  },
})
