import { Cause, Context, Duration, Effect, Layer, Schema, Stream } from "effect"
import { IcnClient, type IcnClientService } from "../client.js"
import { HardwareSnapshot as HardwareSnapshotSchema } from "@magnitudedev/icn-protocol/schemas"
import {
  makeIcnObservedState,
  type IcnObservedSnapshot,
  type IcnObservedState,
} from "../observed-state.js"
import { IcnEvents, refreshOnIcnEvents } from "../events/index.js"

type HardwareReadError = Effect.Effect.Error<ReturnType<IcnClientService["system"]["getHardware"]>>

export interface IcnHardwareService extends IcnObservedState<HardwareSnapshotSchema, HardwareReadError> {
  /**
   * Hardware changes that can alter model-assessment evidence. Live availability
   * changes remain on `changes` for admission and presentation consumers.
   */
  readonly assessmentChanges: Stream.Stream<IcnObservedSnapshot<HardwareSnapshotSchema>>
}

export class IcnHardware extends Context.Tag("@magnitudedev/icn/IcnHardware")<
  IcnHardware,
  IcnHardwareService
>() {}

export interface IcnHardwareOptions {
  readonly refreshInterval?: Duration.DurationInput
}

export const makeIcnHardware = (
  options: IcnHardwareOptions = {},
): Layer.Layer<IcnHardware, HardwareReadError, IcnClient | IcnEvents> =>
  Layer.scoped(
    IcnHardware,
    Effect.gen(function* () {
      const client = yield* IcnClient
      const events = yield* IcnEvents
      const invalidations = yield* events.subscribe
      const read = client.system.getHardware({})
      const initial = yield* read
      const observed = yield* makeIcnObservedState(
        initial,
        read,
        Schema.equivalence(HardwareSnapshotSchema),
        { initiallyInitialized: true },
      )

      // Instance transitions alter the allocatable-memory view immediately.
      // Polling remains necessary for memory consumed by unrelated processes.
      yield* refreshOnIcnEvents(
        invalidations,
        new Set(["hardware"]),
        observed.refresh,
        "ICN hardware snapshot",
        options.refreshInterval ?? "2 seconds",
      ).pipe(Effect.forkScoped)

      yield* observed.refresh.pipe(
        Effect.tapError((error) => Effect.logWarning("Unable to refresh ICN hardware snapshot").pipe(
          Effect.annotateLogs({ cause: Cause.pretty(Cause.fail(error)) }),
        )),
        Effect.option,
        Effect.delay(options.refreshInterval ?? "2 seconds"),
        Effect.forever,
        Effect.forkScoped,
      )

      const assessmentChanges = observed.changes.pipe(Stream.changesWith((previous, next) =>
        previous.state.native_build === next.state.native_build
        && previous.state.topology_fingerprint === next.state.topology_fingerprint
        && previous.state.system_memory.physical_capacity_bytes
          === next.state.system_memory.physical_capacity_bytes
        && previous.state.system_memory.assess_reserve_bytes
          === next.state.system_memory.assess_reserve_bytes
        && previous.state.enabled_backends.length === next.state.enabled_backends.length
        && previous.state.enabled_backends.every(
          (backend, index) => backend === next.state.enabled_backends[index],
        )))

      return IcnHardware.of({
        ...observed,
        assessmentChanges,
      })
    }),
  )
