import { Context, Effect, Layer } from "effect"

export interface LocalModelConfigurationCoordinatorApi {
  readonly exclusive: <A, E, R>(operation: Effect.Effect<A, E, R>) => Effect.Effect<A, E, R>
}

export class LocalModelConfigurationCoordinator extends Context.Tag(
  "LocalModelConfigurationCoordinator",
)<LocalModelConfigurationCoordinator, LocalModelConfigurationCoordinatorApi>() {}

export const LocalModelConfigurationCoordinatorLive = Layer.effect(
  LocalModelConfigurationCoordinator,
  Effect.makeSemaphore(1).pipe(Effect.map((semaphore) =>
    LocalModelConfigurationCoordinator.of({
      exclusive: (operation) => semaphore.withPermits(1)(operation),
    }))),
)
