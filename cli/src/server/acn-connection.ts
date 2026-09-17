import { desktopServiceOrigin, startDesktopApplication } from "./application"
import { FetchHttpClient } from "@effect/platform"
import { MagnitudeClient } from "@magnitudedev/sdk"
import { makeFirstPartyConnection } from "@magnitudedev/client-common"
import { Effect, Layer } from "effect"

/** Connection whose startup requires an already-usable service. */
export const existingAcnConnection = makeFirstPartyConnection(
  MagnitudeClient.layer({ origin: desktopServiceOrigin, autoStart: false }).pipe(Layer.provide(FetchHttpClient.layer)),
)

/** A headless request may initially start the desktop in the background, never a separate daemon. */
export const headlessAcnConnection = Effect.flatMap(startDesktopApplication, () => makeFirstPartyConnection(
  MagnitudeClient.layer({ origin: desktopServiceOrigin, autoStart: false }).pipe(Layer.provide(FetchHttpClient.layer)),
))
