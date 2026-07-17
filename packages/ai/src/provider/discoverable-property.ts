import { Schema } from "effect"
import { FSM } from "@magnitudedev/utils"

export const ModelDiscoveryOperationIdSchema = Schema.String.pipe(
  Schema.minLength(1),
  Schema.maxLength(256),
  Schema.brand("ModelDiscoveryOperationId"),
)
export type ModelDiscoveryOperationId = typeof ModelDiscoveryOperationIdSchema.Type

export const ModelDiscoveryPhaseSchema = Schema.Literal("queued", "loading", "inspecting")
export type ModelDiscoveryPhase = typeof ModelDiscoveryPhaseSchema.Type

export const ModelPropertyDiscoveryErrorSchema = Schema.Struct({
  code: Schema.String.pipe(Schema.minLength(1)),
  message: Schema.String.pipe(Schema.minLength(1)),
  retryable: Schema.Boolean,
})
export type ModelPropertyDiscoveryError = typeof ModelPropertyDiscoveryErrorSchema.Type

export const ModelPropertyNameSchema = Schema.Literal("vision", "reasoning")
export type ModelPropertyName = typeof ModelPropertyNameSchema.Type

export const ModelPropertyDiscoveryRequestSchema = Schema.Struct({
  providerModelId: Schema.String.pipe(Schema.minLength(1), Schema.maxLength(4096), Schema.brand("ProviderModelId")),
  properties: Schema.Array(ModelPropertyNameSchema).pipe(Schema.minItems(1)),
})
export type ModelPropertyDiscoveryRequest = typeof ModelPropertyDiscoveryRequestSchema.Type

export const defineModelDiscoverableProperty = <A, I, R>(Value: Schema.Schema<A, I, R>) => {
  class Deferred extends Schema.TaggedClass<Deferred>()("Deferred", {}) {}
  class Discovering extends Schema.TaggedClass<Discovering>()("Discovering", {
    operationId: ModelDiscoveryOperationIdSchema,
    phase: ModelDiscoveryPhaseSchema,
  }) {}
  class Cached extends Schema.TaggedClass<Cached>()("Cached", { value: Value }) {}
  class Resolved extends Schema.TaggedClass<Resolved>()("Resolved", { value: Value }) {}
  class Refreshing extends Schema.TaggedClass<Refreshing>()("Refreshing", {
    value: Value,
    operationId: ModelDiscoveryOperationIdSchema,
    phase: ModelDiscoveryPhaseSchema,
  }) {}
  class Failed extends Schema.TaggedClass<Failed>()("Failed", { error: ModelPropertyDiscoveryErrorSchema }) {}

  const SchemaType = Schema.Union(Deferred, Discovering, Cached, Resolved, Refreshing, Failed)
  const Lifecycle = FSM.defineFSM(
    { Deferred, Discovering, Cached, Resolved, Refreshing, Failed },
    {
      Deferred: ["Discovering"],
      Discovering: ["Resolved", "Failed", "Deferred"],
      Cached: ["Refreshing", "Resolved", "Deferred"],
      Resolved: ["Refreshing", "Cached", "Deferred"],
      Refreshing: ["Resolved", "Cached", "Deferred"],
      Failed: ["Discovering", "Deferred"],
    } as const,
  )

  return {
    Schema: SchemaType,
    Lifecycle,
    states: { Deferred, Discovering, Cached, Resolved, Refreshing, Failed },
  } as const
}
