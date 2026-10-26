import { FetchHttpClient } from "@effect/platform"
import { Context, Effect, Layer, Option } from "effect"
import { OperationFailed } from "@magnitudedev/acn-protocol"
import {
  ProviderClient,
  ProviderIdSchema,
  type CloudUsageResponse,
  type ProviderClientShape,
  type UsageQuery,
} from "@magnitudedev/providers/client"
import { MagnitudeStorage } from "@magnitudedev/storage"

const MAGNITUDE_PROVIDER_ID = ProviderIdSchema.make("magnitude")

export interface MagnitudeCloudUsageApi {
  readonly get: (query?: UsageQuery) => Effect.Effect<CloudUsageResponse, OperationFailed>
}

export class MagnitudeCloudUsage extends Context.Tag("MagnitudeCloudUsage")<
  MagnitudeCloudUsage,
  MagnitudeCloudUsageApi
>() {}

const failure = (operation: string) => (cause: unknown): OperationFailed =>
  new OperationFailed({ operation, reason: cause instanceof Error ? cause.message : String(cause) })

export const MagnitudeCloudUsageLive: Layer.Layer<
  MagnitudeCloudUsage,
  never,
  ProviderClient | MagnitudeStorage
> = Layer.effect(MagnitudeCloudUsage, Effect.gen(function* () {
  const client = yield* ProviderClient
  const storage = yield* MagnitudeStorage
  const authenticatedClient: Effect.Effect<ProviderClientShape, OperationFailed> = Effect.gen(function* () {
    const environment = process.env.MAGNITUDE_API_KEY?.trim()
    if (environment) return client
    const stored = yield* storage.auth.get(MAGNITUDE_PROVIDER_ID).pipe(
      Effect.map(Option.fromNullable),
      Effect.orElseSucceed(Option.none),
    )
    if (Option.exists(stored, (auth) => auth.type === "api" && auth.key.trim().length > 0)) return client
    return yield* new OperationFailed({ operation: "get cloud usage", reason: "No Magnitude API key found" })
  })
  return MagnitudeCloudUsage.of({
    // Cloud is disabled.
    get: (_query) => Effect.fail(new OperationFailed({
      operation: "get cloud usage",
      reason: "Magnitude Cloud is disabled",
    })),
    // get: (query) => authenticatedClient.pipe(
    //   Effect.flatMap((authenticated) => authenticated.usage(query).pipe(Effect.provide(FetchHttpClient.layer))),
    //   Effect.mapError(failure("get cloud usage")),
    // ),
  })
}))
