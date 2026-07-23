import { Effect, Layer, Option } from "effect"
import {
  IcnProviderModelResolver,
  type IcnProviderModelResolution,
} from "@magnitudedev/icn/provider"
import { ProviderModelIdSchema } from "@magnitudedev/sdk"
import { LocalProviderOfferings } from "./local-provider-offerings"

export const LocalProviderResolverLive: Layer.Layer<
  IcnProviderModelResolver,
  never,
  LocalProviderOfferings
> = Layer.effect(IcnProviderModelResolver, Effect.gen(function* () {
  const offerings = yield* LocalProviderOfferings
  return IcnProviderModelResolver.of({
    resolve: (providerModelId) => offerings.resolve(providerModelId).pipe(
      Effect.map((offering): Option.Option<IcnProviderModelResolution> => Option.some({
        runtimeModelId: ProviderModelIdSchema.make(offering.configuration.id),
      })),
      Effect.catchAll(() => Effect.succeed(Option.none())),
    ),
  })
}))
