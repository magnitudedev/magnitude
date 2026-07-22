import { Cause, Context, Duration, Effect, Layer, Option, Schema } from "effect"
import { IcnClient, type IcnClientService } from "../client.js"
import { ModelList } from "../generated/schemas.js"
import {
  makeIcnObservedState,
  type IcnObservedState,
} from "../observed-state.js"

type InventoryReadError = Effect.Effect.Error<ReturnType<IcnClientService["models"]["listModels"]>>

export interface IcnInventoryService extends IcnObservedState<ModelList, InventoryReadError> {}

export class IcnInventory extends Context.Tag("@magnitudedev/icn/IcnInventory")<
  IcnInventory,
  IcnInventoryService
>() {}

export interface IcnInventoryOptions {
  readonly reconnectDelay?: Duration.DurationInput
}

export const makeIcnInventory = (
  options: IcnInventoryOptions = {},
): Layer.Layer<IcnInventory, InventoryReadError, IcnClient> =>
  Layer.scoped(
    IcnInventory,
    Effect.gen(function* () {
      const client = yield* IcnClient
      const read = client.models.listModels({})
      const initial = yield* read
      const observed = yield* makeIcnObservedState(
        initial,
        read,
        Schema.equivalence(ModelList),
      )
      // Fetch-backed requests are not reliably interruptible on every runtime.
      // Disconnect the bounded long-poll so layer shutdown never waits on the
      // transport before it can reap the ICN process.
      const observeRevision = (after: Option.Option<number>) =>
        client.runtime
          .observeRuntimeChanges({ urlParams: { after } })
          .pipe(Effect.disconnect)
      const observeChanges = Effect.gen(function* () {
        let revision = (yield* observeRevision(Option.none())).revision
        while (true) {
          const nextRevision = (yield* observeRevision(Option.some(revision))).revision
          if (nextRevision === revision) continue
          revision = nextRevision
          yield* observed.refresh
        }
      })

      yield* observeChanges.pipe(
        Effect.catchAll((error) =>
          Effect.logWarning("ICN runtime change observation disconnected; retrying").pipe(
            Effect.annotateLogs({ cause: Cause.pretty(Cause.fail(error)) }),
            Effect.zipRight(
              Effect.sleep(options.reconnectDelay ?? "250 millis"),
            ),
          ),
        ),
        Effect.forever,
        Effect.forkScoped,
      )

      return IcnInventory.of(observed)
    }),
  )
