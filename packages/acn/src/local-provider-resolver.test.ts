import { Effect, Layer, Option, Stream } from "effect"
import { describe, expect, it } from "vitest"
import type { LocalProviderOffering } from "@magnitudedev/acn-protocol"
import { ProviderModelIdSchema } from "@magnitudedev/sdk"
import { IcnProviderModelResolver } from "@magnitudedev/icn/provider"
import { LocalProviderOfferings } from "./local-provider-offerings"
import { LocalProviderResolverLive } from "./local-provider-resolver"

describe("local provider model resolution", () => {
  it("preserves the canonical provider model identity sent to ICN", async () => {
    const providerModelId = ProviderModelIdSchema.make("gemma-4-26b-a4b-it-qat:gguf:q4")
    const offering = { providerModelId } as LocalProviderOffering
    const offerings = Layer.succeed(LocalProviderOfferings, LocalProviderOfferings.of({
      ready: Effect.succeed(true),
      list: Effect.succeed([offering]),
      changes: Stream.never,
      catalog: Effect.succeed([]),
      state: Effect.succeed({
        entries: [],
        failure: Option.none(),
      }),
      catalogChanges: Stream.never,
      resolve: (requested) => requested === providerModelId
        ? Effect.succeed(offering)
        : Effect.die("unexpected model"),
    }))

    const resolution = await Effect.runPromise(Effect.gen(function* () {
      const resolver = yield* IcnProviderModelResolver
      return yield* resolver.resolve(providerModelId)
    }).pipe(
      Effect.provide(LocalProviderResolverLive.pipe(Layer.provide(offerings))),
    ))

    expect(Option.getOrThrow(resolution).runtimeModelId).toBe(providerModelId)
  })
})
