import { Effect, Layer, Option } from "effect"
import {
  IcnProviderModelResolver,
  type IcnProviderModelResolution,
} from "@magnitudedev/icn/provider"
import { LocalProviderOfferings, localProviderModelId } from "./local-provider-offerings"

export const LocalProviderResolverLive: Layer.Layer<
  IcnProviderModelResolver,
  never,
  LocalProviderOfferings
> = Layer.effect(IcnProviderModelResolver, Effect.gen(function* () {
  const offerings = yield* LocalProviderOfferings
  return IcnProviderModelResolver.of({
    resolve: (providerModelId) => offerings.resolve(providerModelId).pipe(
      Effect.map((offering): Option.Option<IcnProviderModelResolution> => Option.some({
        runtimeModelId: localProviderModelId(offering.configuration.id),
      })),
      Effect.catchAll(() => Effect.succeed(Option.none())),
    ),
  })
}))
