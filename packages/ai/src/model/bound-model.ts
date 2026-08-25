import type { Effect } from "effect"
import type * as HttpClient from "@effect/platform/HttpClient"
import { Prompt } from "../prompt/prompt"
import type { ToolCallId } from "../prompt/ids"
import type { ToolDefinition } from "../tools/tool-definition"
import type { StreamStartFailure } from "../errors/failure"
import type { ModelPreparationObserver, ModelStreamResult } from "./model-spec"

export interface BoundModel<
  TCallOptions,
  TStartFailure = StreamStartFailure,
  TPreparation = never,
> {
  readonly stream: (
    prompt: Prompt,
    tools: readonly ToolDefinition[],
    options?: TCallOptions & { generateToolCallId?: () => ToolCallId },
    onPreparation?: ModelPreparationObserver<TPreparation>,
  ) => Effect.Effect<
    ModelStreamResult,
    TStartFailure,
    HttpClient.HttpClient
  >
}
