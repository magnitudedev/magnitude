import type { Effect } from "effect"
import type { ToolCallId, ToolResultPart } from "@magnitudedev/ai"
import type { HarnessEvent, ToolResult } from "./events"

export interface ExecuteHookContext {
  readonly toolCallId: ToolCallId
  readonly toolName: string
  readonly toolKey: string
  readonly group: string
  readonly input: unknown
}

export type InterceptorDecision =
  | { readonly _tag: "Proceed"; readonly modifiedInput?: unknown }
  | { readonly _tag: "Reject"; readonly rejection: unknown }

export interface HarnessHooks<R = never> {
  readonly beforeExecute?: (ctx: ExecuteHookContext) => Effect.Effect<InterceptorDecision, never, R>
  readonly afterExecute?: (ctx: ExecuteHookContext & { readonly result: ToolResult }) => Effect.Effect<void, never, R>
  readonly onEvent?: (event: HarnessEvent) => Effect.Effect<void, never, R>
  readonly onEmission?: (ctx: {
    readonly toolCallId: ToolCallId
    readonly toolName: string
    readonly toolKey: string
    readonly value: unknown
  }) => Effect.Effect<void, never, R>
  readonly formatResult?: (toolName: string, toolKey: string, result: ToolResult) => readonly ToolResultPart[]
  readonly onResult?: (ctx: {
    readonly toolCallId: ToolCallId
    readonly toolName: string
    readonly toolKey: string
    readonly result: ToolResult
    readonly parts: readonly ToolResultPart[]
  }) => Effect.Effect<void, never, R>
}
