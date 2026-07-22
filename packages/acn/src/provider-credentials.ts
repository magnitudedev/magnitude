import { FetchHttpClient } from "@effect/platform"
import { Context, Effect, Layer, Option } from "effect"
import { SessionOperationFailed, type ProviderAuth, type SessionError } from "@magnitudedev/protocol"
import { ProviderIdSchema, type ProviderId } from "@magnitudedev/sdk"
import { MagnitudeStorage } from "@magnitudedev/storage"
import { ProviderModelCatalog } from "./provider-model-catalog"
import { ProviderClientRegistry } from "./shared-client"

export interface ProviderCredentialsApi {
  readonly update: (providerId: ProviderId, auth: ProviderAuth) => Effect.Effect<void, SessionError>
  readonly get: (providerId: ProviderId) => Effect.Effect<Option.Option<ProviderAuth>, SessionError>
  readonly list: Effect.Effect<ReadonlyMap<ProviderId, ProviderAuth>, SessionError>
}

export class ProviderCredentials extends Context.Tag("ProviderCredentials")<
  ProviderCredentials,
  ProviderCredentialsApi
>() {}

export const ProviderCredentialsLive: Layer.Layer<
  ProviderCredentials,
  never,
  MagnitudeStorage | ProviderClientRegistry | ProviderModelCatalog
> = Layer.effect(ProviderCredentials, Effect.gen(function* () {
  const storage = yield* MagnitudeStorage
  const clients = yield* ProviderClientRegistry
  const catalog = yield* ProviderModelCatalog
  const failure = (operation: string) => (cause: unknown): SessionError => new SessionOperationFailed({
    operation,
    reason: cause instanceof Error ? cause.message : String(cause),
  })
  return ProviderCredentials.of({
    update: (providerId, auth) => storage.auth.set(providerId, auth).pipe(
      Effect.zipRight(clients.refreshAll.pipe(Effect.provide(FetchHttpClient.layer))),
      Effect.tap(() => catalog.refresh(Option.some(providerId))),
      Effect.mapError(failure("update provider credentials")),
    ),
    get: (providerId) => storage.auth.get(providerId).pipe(
      Effect.map(Option.fromNullable),
      Effect.mapError(failure("get provider credentials")),
    ),
    list: storage.auth.loadAll().pipe(
      Effect.map((auths) => new Map(Object.entries(auths).map(([providerId, auth]) => [
        ProviderIdSchema.make(providerId),
        auth,
      ]))),
      Effect.mapError(failure("list provider credentials")),
    ),
  })
}))
