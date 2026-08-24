import type { ChatCompletionProgress } from "./generated/schemas.js"

/** ICN-owned detail carried by the generic AI Preparing phase. */
export type IcnModelPreparation = Exclude<
  ChatCompletionProgress,
  { readonly phase: "generating" }
>
