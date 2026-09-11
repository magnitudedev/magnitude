import { Schema } from "effect"
import {
  NativeChatCompletions,
  Option,
  type BaseCallOptions,
  type BoundModel,
  type ModelSpec,
  type StreamStartFailure,
  type ToolChoice,
} from "@magnitudedev/ai"
import { classifyMiniMaxRejectedResponse } from "./errors"

export type MiniMaxCallOptions = {
  readonly maxTokens?: number
  readonly toolChoice?: ToolChoice
  readonly reasoningEffort?: string
}

export interface MiniMaxCompatibleSpecConfig {
  readonly modelId: string
  readonly endpoint: string
  readonly reasoningMode: "configurable" | "always_on"
}

const configurableReasoning = Option.define(
  Schema.Struct({
    thinking: Schema.Struct({ type: Schema.String }),
    reasoning_split: Schema.Boolean,
  }),
  (reasoningEffort: string) => {
    if (reasoningEffort !== "adaptive" && reasoningEffort !== "disabled") {
      throw new Error(`Unsupported MiniMax-M3 reasoning effort: ${reasoningEffort}`)
    }
    return {
      thinking: { type: reasoningEffort },
      reasoning_split: true,
    }
  },
  "adaptive" as string,
)

const alwaysOnReasoning = Option.define(
  Schema.Struct({ reasoning_split: Schema.Boolean }),
  (reasoningEffort: string) => {
    if (reasoningEffort !== "always_on") {
      throw new Error(`Unsupported MiniMax-M2.7 reasoning effort: ${reasoningEffort}`)
    }
    return { reasoning_split: true }
  },
  "always_on" as string,
)

export const createMiniMaxCompatibleSpec = (
  config: MiniMaxCompatibleSpecConfig,
): ModelSpec<MiniMaxCallOptions> => NativeChatCompletions.model({
  modelId: config.modelId,
  endpoint: config.endpoint,
  options: {
    maxTokens: Option.field("max_completion_tokens", Schema.Number),
    toolChoice: NativeChatCompletions.options.toolChoice,
    reasoningEffort: config.reasoningMode === "configurable"
      ? configurableReasoning
      : alwaysOnReasoning,
  },
  classifyRejectedResponse: classifyMiniMaxRejectedResponse,
})

export const wrapAsBaseModel = <TPreparation = never>(
  internal: BoundModel<MiniMaxCallOptions, StreamStartFailure, TPreparation>,
): BoundModel<BaseCallOptions, StreamStartFailure, TPreparation> => ({
  stream: (prompt, tools, options) => internal.stream(prompt, tools, options),
})
