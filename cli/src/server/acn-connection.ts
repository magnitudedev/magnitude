import { FetchHttpClient } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import {
  AcnInstanceManager,
  makeAcnConnection,
  makeLocalAcnRequireRunningInstanceManager,
  makeLocalAcnStartingInstanceManager,
  type AcnConnection,
} from "@magnitudedev/sdk"
import { BunSqliteDriverLayer } from "@magnitudedev/sdk/bun"
import { Effect, Scope } from "effect"

export const makeAcnConnectionWithInstanceManager = (
  manager: AcnInstanceManager,
): Effect.Effect<AcnConnection, never, Scope.Scope> => makeAcnConnection().pipe(
  Effect.provideService(AcnInstanceManager, manager),
)

const makeObservedAcnConnection = (
  managerEffect: ReturnType<typeof makeLocalAcnRequireRunningInstanceManager>,
): Effect.Effect<AcnConnection, never, Scope.Scope> => Effect.gen(function* () {
  const manager = yield* managerEffect.pipe(
    Effect.provide([BunContext.layer, FetchHttpClient.layer, BunSqliteDriverLayer]),
  )
  return yield* makeAcnConnectionWithInstanceManager(manager)
})

/** Connection whose startup requires an already-usable service. */
export const existingAcnConnection = makeObservedAcnConnection(
  makeLocalAcnRequireRunningInstanceManager(),
)

/** Connection that follows the persistent service launched by `magnitude service start`. */
export const startingAcnConnection = makeObservedAcnConnection(
  makeLocalAcnStartingInstanceManager(),
)
