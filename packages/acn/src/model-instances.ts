import { Context, Effect, Layer, Stream } from "effect"
import {
  Models,
  ModelInstanceIdSchema,
  type ModelInstancesState,
} from "@magnitudedev/acn-protocol"
import { ProviderModelIdSchema, projectInferenceResidency } from "@magnitudedev/sdk"
import { IcnInstances } from "@magnitudedev/icn"
import type { ModelInstancesSnapshot } from "@magnitudedev/icn-protocol/schemas"
import { AcnChanges } from "./changes"

export interface ModelInstancesApi {
  readonly state: Effect.Effect<ModelInstancesState>
  readonly changes: Stream.Stream<ModelInstancesState>
}

export class ModelInstances extends Context.Tag("ModelInstances")<ModelInstances, ModelInstancesApi>() {}

const project = (snapshot: ModelInstancesSnapshot): ModelInstancesState => ({
  instances: snapshot.instances.map((instance) => ({
    instanceId: ModelInstanceIdSchema.make(instance.id),
    modelId: ProviderModelIdSchema.make(instance.modelId),
    residency: projectInferenceResidency(instance),
  })),
})

export const ModelInstancesLive: Layer.Layer<
  ModelInstances,
  never,
  IcnInstances | AcnChanges
> = Layer.scoped(ModelInstances, Effect.gen(function* () {
  const instances = yield* IcnInstances
  const changes = yield* AcnChanges
  const projectedChanges = instances.changes.pipe(Stream.map(project))
  yield* projectedChanges.pipe(
    Stream.runForEach(() => changes.publish({ query: Models.GetInstances.name })),
    Effect.forkScoped,
  )
  return ModelInstances.of({
    state: instances.get.pipe(Effect.map(project)),
    changes: projectedChanges,
  })
}))
