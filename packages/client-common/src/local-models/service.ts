import { useCallback } from "react"
import { Atom, Registry, Result, useAtomSet } from "@effect-atom/atom-react"
import { Context, Effect, Layer } from "effect"
import { Mutation, QueryClient } from "@magnitudedev/effect-query"
import {
  Models,
  type ProviderModelId,
} from "@magnitudedev/sdk"
import { useAgentClient } from "../state/agent-client-context"
import { ClientEffectQuery } from "../state/client-effect-query"
import { localModelsFromCatalog } from "../model-catalog/projection"

const makeLocalModels = Effect.gen(function* () {
  const effectQuery = yield* ClientEffectQuery
  const queryClient = yield* QueryClient.QueryClient
  const registry = yield* Registry.AtomRegistry
  const query = effectQuery.Models.GetCatalog({})
  const install = effectQuery.Models.SyncLocalModel
  const cancelDownload = effectQuery.Models.CancelLocalModelSync
  const dismissDownloadFailure = effectQuery.Models.AcknowledgeLocalModelSyncFailure
  const remove = effectQuery.Models.RemoveLocalModel
  const state = Atom.make((get) => Result.map(get(query).result, localModelsFromCatalog))
  const catalog = Atom.make((get) => Result.map(
    get(state),
    (models) => ({
      ...models,
      models: models.models.filter((model) => model.catalogMembershipState._tag === "InCatalog"),
    }),
  ))
  const provideRegistry = Effect.provideService(Registry.AtomRegistry, registry)

  return {
    state,
    catalog,
    retry: queryClient.invalidate(Models.GetCatalog.match()),
    install: (modelId: ProviderModelId) => Mutation.execute(install, { modelId }).pipe(provideRegistry),
    cancelDownload: (modelId: ProviderModelId) =>
      Mutation.execute(cancelDownload, { modelId }).pipe(
        provideRegistry,
      ),
    dismissDownloadFailure: (modelId: ProviderModelId) =>
      Mutation.execute(dismissDownloadFailure, { modelId }).pipe(
        provideRegistry,
      ),
    remove: (modelId: ProviderModelId) => Mutation.execute(remove, { modelId }).pipe(provideRegistry),
  }
})

export interface LocalModels extends Effect.Effect.Success<typeof makeLocalModels> {}

export type LocalModelsInstallError = Effect.Effect.Error<ReturnType<LocalModels["install"]>>
export type LocalModelsCancelError = Effect.Effect.Error<ReturnType<LocalModels["cancelDownload"]>>

export const LocalModels = Context.GenericTag<LocalModels>("client/LocalModels")

export const LocalModelsLive = Layer.scoped(LocalModels, makeLocalModels)

export function useLocalModelMutations() {
  const client = useAgentClient()
  const install = useAtomSet(client.Models.SyncLocalModel)
  const cancel = useAtomSet(client.Models.CancelLocalModelSync)
  const dismissFailure = useAtomSet(client.Models.AcknowledgeLocalModelSyncFailure)
  const removeModel = useAtomSet(client.Models.RemoveLocalModel)

  return {
    install: useCallback((modelId: ProviderModelId) => {
      install({ modelId })
    }, [install]),
    cancel: useCallback((modelId: ProviderModelId) => {
      cancel({ modelId })
    }, [cancel]),
    dismissFailure: useCallback((modelId: ProviderModelId) => {
      dismissFailure({ modelId })
    }, [dismissFailure]),
    remove: useCallback((modelId: ProviderModelId) => {
      removeModel({ modelId })
    }, [removeModel]),
  }
}
