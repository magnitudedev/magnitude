import { Schema } from "effect"
import type { ImagePart } from "../prompt/parts"

export const ProviderModelCapabilitiesSchema = Schema.Struct({
  vision: Schema.optional(Schema.Boolean),
})
export type ProviderModelCapabilities = Schema.Schema.Type<typeof ProviderModelCapabilitiesSchema>

export interface ImagePlaceholderConfig {
  readonly enabled: boolean
  readonly format?: (part: ImagePart) => string
}
