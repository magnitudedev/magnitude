import { Context, Effect, Layer } from "effect"
import type { CatalogComponent, CatalogFormModelId, LocalModelMutationFailed, LocalModelSyncOutcome } from "@magnitudedev/acn-protocol"
import { IcnCatalog, IcnCatalogInstallations, IcnClient } from "@magnitudedev/icn"
import { LocalModelRemovals } from "./local-model-removals"
import { icnCommandFailure } from "./icn-command-failure"

/** Installing or updating a catalog model, whichever command asked for it. */
export interface CatalogModelSyncApi {
  readonly installed: (modelId: CatalogFormModelId) => Effect.Effect<boolean>
  readonly sync: (modelId: CatalogFormModelId) => Effect.Effect<{ readonly outcome: LocalModelSyncOutcome }, LocalModelMutationFailed>
  /** Fetches only one component the model lacks; false when it is already here. */
  readonly downloadComponent: (modelId: CatalogFormModelId, component: CatalogComponent) =>
    Effect.Effect<{ readonly started: boolean }, LocalModelMutationFailed>
}
export class CatalogModelSync extends Context.Tag("CatalogModelSync")<CatalogModelSync, CatalogModelSyncApi>() {}

export const CatalogModelSyncLive: Layer.Layer<
  CatalogModelSync,
  never,
  IcnClient | IcnCatalog | IcnCatalogInstallations | LocalModelRemovals
> = Layer.effect(CatalogModelSync, Effect.gen(function* () {
  const client = yield* IcnClient
  const catalog = yield* IcnCatalog
  const installations = yield* IcnCatalogInstallations
  const removals = yield* LocalModelRemovals
  const installed = (modelId: CatalogFormModelId) => catalog.get.pipe(Effect.map(({ state }) =>
    state.models.some((model) => model.id === modelId && model.localState._tag === "Installed")))
  return CatalogModelSync.of({
    installed,
    sync: (modelId) => Effect.gen(function* () {
      yield* removals.forget(modelId)
      const wasInstalled = yield* installed(modelId)
      const admission = yield* client.catalog.installCatalogModel({ path: { model_id: modelId } }).pipe(
        Effect.mapError((cause) => icnCommandFailure("sync", cause)),
      )
      yield* Effect.all([catalog.refresh, installations.refresh], { discard: true }).pipe(Effect.ignore)
      const outcome: LocalModelSyncOutcome = admission._tag === "Current" ? "AlreadyCurrent" : wasInstalled ? "Updating" : "Installing"
      return { outcome }
    }),
    downloadComponent: (modelId, component) => Effect.gen(function* () {
      const admission = yield* client.catalog.installCatalogComponent({ path: { model_id: modelId, component } }).pipe(
        Effect.mapError((cause) => icnCommandFailure("download_component", cause)),
      )
      yield* Effect.all([catalog.refresh, installations.refresh], { discard: true }).pipe(Effect.ignore)
      return { started: admission._tag !== "Current" }
    }),
  })
}))
