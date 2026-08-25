import { Context, Effect, Layer, Option, Ref, Stream } from "effect"
import {
  CustomEndpointNameSchema,
  type CustomEndpointDeclarations,
} from "@magnitudedev/storage"
import {
  ProviderModelIdSchema,
  customEndpointProviderId,
  type ProviderId,
  type ProviderModelId,
} from "@magnitudedev/sdk"
import { CustomEndpoints } from "./custom-endpoints"
import { ModelSlotController } from "./model-slot-controller"
import { ProviderModelCatalog } from "./provider-model-catalog"
import { ProviderClientRegistry } from "./shared-client"

export class CustomEndpointReconciler extends Context.Tag("CustomEndpointReconciler")<
  CustomEndpointReconciler,
  Record<never, never>
>() {}

export interface CustomEndpointModelIdentity {
  readonly providerId: ProviderId
  readonly providerModelId: ProviderModelId
}

export const removedCustomEndpointModels = (
  previous: CustomEndpointDeclarations,
  next: CustomEndpointDeclarations,
): readonly CustomEndpointModelIdentity[] => {
  const removed: CustomEndpointModelIdentity[] = []
  for (const [name, declaration] of Object.entries(previous)) {
    const endpointName = CustomEndpointNameSchema.make(name)
    const providerId = customEndpointProviderId(endpointName)
    const replacement = next[endpointName]
    for (const modelId of Object.keys(declaration.models)) {
      if (replacement === undefined || !(modelId in replacement.models)) {
        removed.push({
          providerId,
          providerModelId: ProviderModelIdSchema.make(modelId),
        })
      }
    }
  }
  return removed
}

export const CustomEndpointReconcilerLive: Layer.Layer<
  CustomEndpointReconciler,
  never,
  CustomEndpoints | ModelSlotController | ProviderModelCatalog | ProviderClientRegistry
> = Layer.scoped(CustomEndpointReconciler, Effect.gen(function* () {
  const endpoints = yield* CustomEndpoints
  const slots = yield* ModelSlotController
  const catalog = yield* ProviderModelCatalog
  const clients = yield* ProviderClientRegistry
  const previous = yield* Ref.make(yield* endpoints.get)
  const lock = yield* Effect.makeSemaphore(1)

  const reconcile = (next: CustomEndpointDeclarations) => lock.withPermits(1)(Effect.gen(function* () {
    const current = yield* Ref.get(previous)
    const removed = removedCustomEndpointModels(current, next)
    if (removed.length > 0) {
      const snapshot = (yield* slots.state).slots
      for (const slot of [snapshot.primary, snapshot.secondary]) {
        if (slot._tag !== "Unassigned"
          && removed.some((identity) =>
            identity.providerId === slot.selection.providerId
            && identity.providerModelId === slot.selection.providerModelId)) {
          yield* slots.updateModelSlot(slot.slotId, Option.none())
        }
      }
    }
    yield* clients.refreshAll
    yield* catalog.refresh(Option.none())
    yield* Ref.set(previous, next)
  })).pipe(
    Effect.catchAllCause((cause) => Effect.logError("Unable to reconcile custom endpoints").pipe(
      Effect.annotateLogs({ cause: String(cause).slice(0, 2_000) }),
    )),
  )

  yield* endpoints.changes.pipe(
    Stream.runForEach(reconcile),
    Effect.forkScoped,
  )

  return CustomEndpointReconciler.of({})
}))
