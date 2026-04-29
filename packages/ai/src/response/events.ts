import type { ToolCallId } from "../prompt/ids"
import type { JsonValue } from "../prompt/parts"
import type { ResponseUsage } from "./usage"

export type ResponseStreamEvent =
  | { readonly _tag: "thought_start"; readonly level: "low" | "medium" | "high" }
  | { readonly _tag: "thought_delta"; readonly text: string }
  | { readonly _tag: "thought_end" }
  | { readonly _tag: "message_start" }
  | { readonly _tag: "message_delta"; readonly text: string }
  | { readonly _tag: "message_end" }
  | { readonly _tag: "tool_call_start"; readonly toolCallId: ToolCallId; readonly toolName: string }
  | { readonly _tag: "tool_call_field_start"; readonly toolCallId: ToolCallId; readonly path: readonly string[] }
  | { readonly _tag: "tool_call_field_delta"; readonly toolCallId: ToolCallId; readonly path: readonly string[]; readonly delta: string }
  | { readonly _tag: "tool_call_field_end"; readonly toolCallId: ToolCallId; readonly path: readonly string[]; readonly value: JsonValue }
  | { readonly _tag: "tool_call_end"; readonly toolCallId: ToolCallId }
  | {
      readonly _tag: "response_done"
      readonly reason: string
      readonly usage?: ResponseUsage
    }
