import { FetchHttpClient } from "@effect/platform"
import { Context, Effect, Layer, Option } from "effect"
import { SessionOperationFailed, type SessionError } from "@magnitudedev/protocol"
import {
  ProviderClient,
  ProviderIdSchema,
  type CloudUsageResponse,
  type ProviderClientShape,
  type UsageQuery,
} from "@magnitudedev/sdk"
import { MagnitudeStorage } from "@magnitudedev/storage"

const MAGNITUDE_PROVIDER_ID = ProviderIdSchema.make("magnitude")

export interface MagnitudeCloudUsageApi {
  readonly get: (query?: UsageQuery) => Effect.Effect<CloudUsageResponse, SessionError>
}

export class MagnitudeCloudUsage extends Context.Tag("MagnitudeCloudUsage")<
  MagnitudeCloudUsage,
  MagnitudeCloudUsageApi
>() {}

const failure = (operation: string) => (cause: unknown): SessionError =>
  new SessionOperationFailed({ operation, reason: cause instanceof Error ? cause.message : String(cause) })

export const MagnitudeCloudUsageLive: Layer.Layer<
  MagnitudeCloudUsage,
  never,
  ProviderClient | MagnitudeStorage
> = Layer.effect(MagnitudeCloudUsage, Effect.gen(function* () {
  const client = yield* ProviderClient
  const storage = yield* MagnitudeStorage
  const authenticatedClient: Effect.Effect<ProviderClientShape, SessionError> = Effect.gen(function* () {
    const environment = process.env.MAGNITUDE_API_KEY?.trim()
    if (environment) return client
    const stored = yield* storage.auth.get(MAGNITUDE_PROVIDER_ID).pipe(
      Effect.map(Option.fromNullable),
      Effect.orElseSucceed(Option.none),
    )
    if (Option.exists(stored, (auth) => auth.type === "api" && auth.key.trim().length > 0)) return client
    return yield* new SessionOperationFailed({ operation: "get cloud usage", reason: "No Magnitude API key found" })
  })
  return MagnitudeCloudUsage.of({
    get: (query) => authenticatedClient.pipe(
      Effect.flatMap((authenticated) => authenticated.usage(query).pipe(Effect.provide(FetchHttpClient.layer))),
      Effect.mapError(failure("get cloud usage")),
    ),
  })
}))
