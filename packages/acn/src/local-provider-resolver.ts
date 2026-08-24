import { Effect, Layer, Option } from "effect"
import {
  IcnProviderModelResolver,
  type IcnProviderModelResolution,
} from "@magnitudedev/icn/provider"
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
        runtimeModelId: offering.providerModelId,
      })),
      Effect.catchAll(() => Effect.succeed(Option.none())),
    ),
  })
}))
