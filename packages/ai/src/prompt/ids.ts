import { init } from "@paralleldrive/cuid2"
import { Schema } from "effect"

export type ToolCallId = string & { readonly __brand: "ToolCallId" }
export type ProviderToolCallId = string & { readonly __brand: "ProviderToolCallId" }

export const ToolCallIdSchema = Schema.declare<ToolCallId>(
  (input): input is ToolCallId => typeof input === "string",
)
export const ProviderToolCallIdSchema = Schema.declare<ProviderToolCallId>(
  (input): input is ProviderToolCallId => typeof input === "string",
)

/** Default generator — produces a fresh cuid2 (8 chars). Callable as `() => ToolCallId`. */
export const createToolCallId = (() => {
  const fn = init({ length: 8 })
  return (): ToolCallId => fn() as ToolCallId
})()
