import { Cause, Context, Effect, Layer, Option, PubSub, Schedule, Stream } from "effect"
import type { Duration, Scope } from "effect"
import type {
  InferenceResourceInvalidation,
  InferenceResourceTopic,
} from "@magnitudedev/icn-protocol/schemas"
import { IcnClient, type IcnClientService } from "../client.js"

type EventsWatchError = Effect.Effect.Error<
  ReturnType<IcnClientService["system"]["watchInferenceEvents"]>
>

export type IcnResourceEvent =
  | { readonly _tag: "Invalidated"; readonly invalidation: InferenceResourceInvalidation }
  | { readonly _tag: "Reconnected" }

export interface IcnEventsService {
  /** Establishes a buffered subscriber before its caller reads a snapshot. */
  readonly subscribe: Effect.Effect<Stream.Stream<IcnResourceEvent>, never, Scope.Scope>
}

export class IcnEvents extends Context.Tag("@magnitudedev/icn/IcnEvents")<
  IcnEvents,
  IcnEventsService
>() {}

export interface IcnEventsOptions {
  readonly retryInterval?: Duration.DurationInput
}

/** One private ICN invalidation connection shared by every ACN inference projection. */
export const makeIcnEvents = (
  options: IcnEventsOptions = {},
): Layer.Layer<IcnEvents, EventsWatchError, IcnClient> => Layer.scoped(
  IcnEvents,
  Effect.gen(function* () {
    const client = yield* IcnClient
    const retryInterval = options.retryInterval ?? "1 second"
    const events = yield* Effect.acquireRelease(
      PubSub.unbounded<IcnResourceEvent>(),
      PubSub.shutdown,
    )
    const input = { urlParams: { topics: Option.none<string>() } } as const
    const initialWatch = yield* client.system.watchInferenceEvents(input)
    const admit = client.system.watchInferenceEvents(input).pipe(
      Effect.tapError((error) => Effect.logWarning("Unable to reconnect ICN inference event stream").pipe(
        Effect.annotateLogs({ cause: Cause.pretty(Cause.fail(error)) }),
      )),
      Effect.retry(Schedule.spaced(retryInterval)),
    )
    const consume = (watch: typeof initialWatch) => watch.events.pipe(
      Stream.runForEach((invalidation) => PubSub.publish(events, {
        _tag: "Invalidated",
        invalidation,
      })),
    )

    yield* Effect.iterate(initialWatch, {
      while: () => true,
      body: (watch) => consume(watch).pipe(
        Effect.catchAll((error) => Effect.logWarning("ICN inference event stream terminated").pipe(
          Effect.annotateLogs({ cause: Cause.pretty(Cause.fail(error)) }),
        )),
        Effect.zipRight(Effect.sleep(retryInterval)),
        Effect.zipRight(admit),
        Effect.tap(() => PubSub.publish(events, { _tag: "Reconnected" })),
      ),
    }).pipe(Effect.forkScoped)

    return IcnEvents.of({
      subscribe: Effect.map(PubSub.subscribe(events), Stream.fromQueue),
    })
  }),
)

export const IcnEventsLive = makeIcnEvents()

export const refreshOnIcnEvents = <E>(
  events: Stream.Stream<IcnResourceEvent>,
  topics: ReadonlySet<InferenceResourceTopic>,
  refresh: Effect.Effect<void, E>,
  label: string,
  retryInterval: Duration.DurationInput = "1 second",
) => events.pipe(
  Stream.filter((event) => event._tag === "Reconnected"
    || topics.has(event.invalidation.topic)),
  // Every signal causes the same complete authoritative reread. Drain the
  // shared PubSub independently of that I/O and retain only the newest signal
  // while a refresh is in flight, rather than accumulating redundant download
  // or instance progress invalidations without bound.
  Stream.buffer({ capacity: 1, strategy: "sliding" }),
  Stream.runForEach(() => refresh.pipe(
    Effect.tapError((error) => Effect.logWarning(`Unable to refresh ${label}`).pipe(
      Effect.annotateLogs({ cause: Cause.pretty(Cause.fail(error)) }),
    )),
    Effect.retry(Schedule.spaced(retryInterval)),
  )),
)
