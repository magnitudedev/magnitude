import { Context, Duration, Effect, Layer, Stream } from "effect"
import {
  IcnApiClient,
  type IcnApiClient as IcnApiClientService,
} from "../generated/client.js"
import type { ModelList } from "../generated/schemas.js"
import {
  makeIcnObservedState,
  type IcnObservedState,
} from "../observed-state.js"

type InventoryReadError = Effect.Effect.Error<ReturnType<IcnApiClientService["models"]["listModels"]>>

export interface IcnInventoryService extends IcnObservedState<ModelList, InventoryReadError> {
  readonly getModel: IcnApiClientService["models"]["getModel"]
  readonly configureModelServing: IcnApiClientService["models"]["configureModelServing"]
  readonly downloadModel: IcnApiClientService["models"]["downloadModel"]
  readonly loadModel: IcnApiClientService["models"]["loadModel"]
  readonly unloadModel: IcnApiClientService["models"]["unloadModel"]
  readonly deleteModel: IcnApiClientService["models"]["deleteModel"]
  readonly observeChatAdmission: <A, E, R>(effect: Effect.Effect<A, E, R>) => Effect.Effect<A, E, R>
}

export class IcnInventory extends Context.Tag("@magnitudedev/icn/IcnInventory")<
  IcnInventory,
  IcnInventoryService
>() {}

export interface IcnInventoryOptions {
  readonly idleRefreshInterval?: Duration.DurationInput
  readonly activeRefreshInterval?: Duration.DurationInput
}

export const makeIcnInventory = (
  options: IcnInventoryOptions = {},
): Layer.Layer<IcnInventory, InventoryReadError, IcnApiClient> =>
  Layer.scoped(
    IcnInventory,
    Effect.gen(function* () {
      const client = yield* IcnApiClient
      const read = client.models.listModels({})
      const initial = yield* read
      const observed = yield* makeIcnObservedState(initial, read)
      const refreshIgnoringFailure = observed.refresh.pipe(
        Effect.catchAll((cause) => Effect.logWarning("Unable to refresh ICN model inventory").pipe(
          Effect.annotateLogs({ cause: String(cause) }),
        )),
      )

      yield* refreshIgnoringFailure.pipe(
        Effect.delay(options.idleRefreshInterval ?? "5 seconds"),
        Effect.forever,
        Effect.forkScoped,
      )

      const refreshEvents = <A, E, R>(events: Stream.Stream<A, E, R>): Stream.Stream<A, E, R> =>
        events.pipe(
          Stream.tap(() => refreshIgnoringFailure),
          Stream.ensuring(refreshIgnoringFailure),
        )

      const observeChatAdmission: IcnInventoryService["observeChatAdmission"] = (effect) =>
        Effect.scoped(Effect.gen(function* () {
          yield* refreshIgnoringFailure
          yield* refreshIgnoringFailure.pipe(
            Effect.delay(options.activeRefreshInterval ?? "150 millis"),
            Effect.forever,
            Effect.forkScoped,
          )
          return yield* effect.pipe(Effect.ensuring(refreshIgnoringFailure))
        }))

      return IcnInventory.of({
        ...observed,
        getModel: client.models.getModel,
        configureModelServing: (request) => client.models.configureModelServing(request).pipe(
          Effect.tap(() => refreshIgnoringFailure),
        ),
        downloadModel: (request) => client.models.downloadModel(request).pipe(
          Effect.map((response) => ({
            ...response,
            events: refreshEvents(response.events),
          })),
        ),
        loadModel: (request) => client.models.loadModel(request).pipe(
          Effect.map((response) => ({
            ...response,
            events: refreshEvents(response.events),
          })),
        ),
        unloadModel: (request) => client.models.unloadModel(request).pipe(
          Effect.tap(() => refreshIgnoringFailure),
        ),
        deleteModel: (request) => client.models.deleteModel(request).pipe(
          Effect.tap(() => refreshIgnoringFailure),
        ),
        observeChatAdmission,
      })
    }),
  )
