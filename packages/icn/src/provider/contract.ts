import { FiberRef, Option, Schema } from "effect"
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

export interface ModelInstanceBinding {
  readonly instanceId: string
  readonly configurationId: string
}

/**
 * Exact native target for the current local-provider request. ACN installs the
 * binding in the request fiber after the slot's canonical instance is Ready;
 * the provider reads it when constructing the ICN request.
 */
export const CurrentModelInstance = FiberRef.unsafeMake<
  Option.Option<ModelInstanceBinding>
>(Option.none())
