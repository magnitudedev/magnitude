import type { Effect } from "effect"
import type { JsonValue } from "@magnitudedev/utils/schema"
import type { ToolCallId } from "@magnitudedev/ai"
import type { HarnessEvent, ToolError, ToolResult } from "./events"

export interface ExecuteHookContext {
  readonly toolCallId: ToolCallId
  readonly toolName: string
  readonly toolKey: string
  readonly input: unknown
}

export type InterceptorDecision<TDenial extends JsonValue = JsonValue> =
  | { readonly _tag: "Proceed"; readonly modifiedInput?: unknown }
  | { readonly _tag: "Deny"; readonly denial: TDenial }

export interface HarnessHooks<R = never, TDenial extends JsonValue = JsonValue> {
  readonly beforeExecute?: (ctx: ExecuteHookContext) => Effect.Effect<InterceptorDecision<TDenial>, never, R>
  readonly afterExecute?: (ctx: ExecuteHookContext & { readonly result: ToolResult<JsonValue, ToolError, TDenial> }) => Effect.Effect<void, never, R>
  readonly onEvent?: (event: HarnessEvent) => Effect.Effect<void, never, R>
  readonly onEmission?: (ctx: {
    readonly toolCallId: ToolCallId
    readonly toolName: string
    readonly toolKey: string
    readonly value: unknown
  }) => Effect.Effect<void, never, R>
}
