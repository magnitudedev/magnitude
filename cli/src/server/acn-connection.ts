import { desktopApplication, desktopServiceOrigin, localServiceClient } from "./application"
import { makeFirstPartyConnection } from "@magnitudedev/client-common"
import { Effect, Schema } from "effect"

export class NoServiceRunning extends Schema.TaggedError<NoServiceRunning>()("NoServiceRunning", {}) {
  override get message() { return "No Magnitude service is running. Open the Magnitude desktop app or run `magnitude serve`." }
}

/** Connection whose startup requires an already-usable service. */
export const existingAcnConnection = desktopApplication.observe.pipe(
  Effect.catchTag("ApplicationControlUnavailable", () => Effect.fail(new NoServiceRunning())),
  Effect.zipRight(makeFirstPartyConnection(localServiceClient(desktopServiceOrigin))),
)
