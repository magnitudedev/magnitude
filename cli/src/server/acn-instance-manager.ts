import { FetchHttpClient } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import {
  BunDetachedChildProcessSpawner,
  ChildProcessSpawner,
  makeLocalAcnInstanceManager,
  SDK_ACN_TARGET,
} from "@magnitudedev/sdk"
import { BunSqliteDriverLayer } from "@magnitudedev/sdk/bun"
import { Array as Arr, Effect, Option } from "effect"

export interface BootstrappingAcnInstanceManagerOptions {
  readonly launchCommand: Option.Option<Arr.NonEmptyReadonlyArray<string>>
  readonly debug: boolean
}

const provideLocalAcnDependencies = <A, E, R>(effect: Effect.Effect<A, E, R>) => effect.pipe(
  Effect.provideService(ChildProcessSpawner, BunDetachedChildProcessSpawner),
  Effect.provide([BunContext.layer, FetchHttpClient.layer, BunSqliteDriverLayer]),
)

export const makeBootstrappingAcnInstanceManager = (
  options: BootstrappingAcnInstanceManagerOptions,
) => makeLocalAcnInstanceManager({
  ...(options.debug ? { debug: true } : {}),
  ...Option.match(options.launchCommand, {
    onNone: () => ({}),
    onSome: (command) => ({
      launchOverride: {
        target: SDK_ACN_TARGET,
        command,
      },
    }),
  }),
}).pipe(provideLocalAcnDependencies)

export const stopLocalAcn = Effect.scoped(
  makeLocalAcnInstanceManager().pipe(
    provideLocalAcnDependencies,
    Effect.flatMap((manager) => manager.stop),
  ),
)
