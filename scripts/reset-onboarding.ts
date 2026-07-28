import * as BunContext from "@effect/platform-bun/BunContext"
import * as BunRuntime from "@effect/platform-bun/BunRuntime"
import {
  GlobalStorageLive,
  makeConfigStorage,
} from "@magnitudedev/storage"
import { Console, Effect, Option } from "effect"

const resetOnboarding = Effect.gen(function* () {
  const config = yield* makeConfigStorage()

  yield* config.update((current) => ({
    ...current,
    onboarding: Option.none(),
  }))

  yield* Console.log("Onboarding state reset.")
})

BunRuntime.runMain(
  resetOnboarding.pipe(
    Effect.provide(GlobalStorageLive),
    Effect.provide(BunContext.layer),
  ),
)
