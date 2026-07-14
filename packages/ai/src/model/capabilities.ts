import type { ImagePart } from "../prompt/parts"

export interface ProviderModelCapabilities {
  readonly vision: boolean
  readonly toolCalls: boolean
  readonly structuredOutput: boolean
  readonly grammar: boolean
  readonly toolChoiceModes: readonly ("auto" | "none" | "required" | "named")[]
}

export interface ImagePlaceholderConfig {
  readonly enabled: boolean
  readonly format?: (part: ImagePart) => string
}
