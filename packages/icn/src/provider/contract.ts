import { Schema } from "effect"
import { ProviderModelFields } from "@magnitudedev/ai"

export const LocalProviderId = Schema.Literal("local").pipe(Schema.brand("ProviderId"))

export const LocalModelInfoSchema = Schema.Struct({
  ...ProviderModelFields,
  providerId: LocalProviderId,
}).pipe(Schema.filter((model) => {
  const reasoning = model.properties.reasoning
  return reasoning._tag !== "Cached"
    && reasoning._tag !== "Resolved"
    && reasoning._tag !== "Refreshing"
    || reasoning.value.includes(model.defaultReasoningEffort)
}, { message: () => "Discovered reasoning efforts must contain defaultReasoningEffort" }))

export type LocalModelInfo = Schema.Schema.Type<typeof LocalModelInfoSchema>
