import { FetchHttpClient } from "@effect/platform"
import { type AcnInstanceManager, makeServiceStarter } from "@magnitudedev/daemon-management"
import { MagnitudeClient, MagnitudeServiceStarter } from "@magnitudedev/sdk"
import { makeFirstPartyConnection, type FirstPartyConnection } from "@magnitudedev/client-common"
import { Effect, Layer, Scope, Stream } from "effect"

export const makeAcnConnectionWithInstanceManager = (
  manager: AcnInstanceManager,
): Effect.Effect<FirstPartyConnection, never, Scope.Scope> => makeFirstPartyConnection(
  MagnitudeClient.layer().pipe(Layer.provide([
    FetchHttpClient.layer,
    Layer.succeed(MagnitudeServiceStarter, makeServiceStarter(manager)),
  ])),
)

/** Connection whose startup requires an already-usable service. */
export const existingAcnConnection = makeFirstPartyConnection(
  MagnitudeClient.layer({ autoStart: false }).pipe(Layer.provide(FetchHttpClient.layer)),
)

/** Connection that follows the persistent service launched by `magnitude service start`. */
export const startingAcnConnection = makeFirstPartyConnection(MagnitudeClient.layer().pipe(
  Layer.provide([
    FetchHttpClient.layer,
    Layer.succeed(MagnitudeServiceStarter, { start: Stream.empty }),
  ]),
))
