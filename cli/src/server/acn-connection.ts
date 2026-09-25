import { desktopApplication, desktopServiceOrigin } from "./application"
import { FetchHttpClient } from "@effect/platform"
import { MagnitudeClient } from "@magnitudedev/sdk"
import { makeFirstPartyConnection } from "@magnitudedev/client-common"
import { Effect, Layer, Schema } from "effect"

export class NoServiceRunning extends Schema.TaggedError<NoServiceRunning>()("NoServiceRunning", {}) {
  override get message() { return "No Magnitude service is running. Open the Magnitude desktop app or run `magnitude serve`." }
}

/** Connection whose startup requires an already-usable service. */
export const existingAcnConnection = desktopApplication.observe.pipe(
  Effect.catchTag("ApplicationControlUnavailable", () => Effect.fail(new NoServiceRunning())),
  Effect.zipRight(makeFirstPartyConnection(
    MagnitudeClient.layer({ origin: desktopServiceOrigin, autoStart: false }).pipe(Layer.provide(FetchHttpClient.layer)),
  )),
)
