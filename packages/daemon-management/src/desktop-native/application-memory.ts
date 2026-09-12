import { createRequire } from "node:module"
import { ApplicationMemorySample, type ApplicationMemoryObservation } from "@magnitudedev/sdk/desktop-host"
import { Clock, Context, Effect, Layer, Option, Schedule, Schema, Stream } from "effect"

export class ApplicationMemoryUnavailable extends Schema.TaggedError<ApplicationMemoryUnavailable>()("ApplicationMemoryUnavailable", { message: Schema.String }) {}
export interface ApplicationMemory {
  readonly read: Effect.Effect<ApplicationMemoryObservation>
}
export const ApplicationMemory = Context.GenericTag<ApplicationMemory>("@magnitudedev/daemon-management/ApplicationMemory")
export const applicationMemoryLayerFromLoader = (load: () => unknown) => Layer.effect(ApplicationMemory, Effect.gen(function* () {
  // A missing/unsupported observer is a telemetry failure, never an application startup failure.
  const binding = yield* Effect.try({
    try: () => {
      const value = load() as { applicationMemory?: unknown }
      if (typeof value?.applicationMemory !== "function") throw new Error("Memory observer missing")
      return value.applicationMemory as () => Promise<unknown>
    },
    catch: () => new ApplicationMemoryUnavailable({ message: "Memory measurement is unavailable in this build." }),
  }).pipe(Effect.either)
  const semaphore = yield* Effect.makeSemaphore(1)
  const read = Effect.gen(function* () {
    if (binding._tag === "Left") return yield* binding.left
    const measured = yield* Effect.tryPromise({
      try: () => binding.right(),
      catch: () => new ApplicationMemoryUnavailable({ message: "Memory reading unavailable. Magnitude will try again shortly." }),
    }).pipe(Effect.flatMap(Schema.decodeUnknown(ApplicationMemorySample)))
    return { _tag: "Measured" as const, ...measured, measuredAt: yield* Clock.currentTimeMillis }
  }).pipe(
    // Finish the bounded native observation before releasing its permit, even if the UI disconnects.
    Effect.uninterruptible, semaphore.withPermits(1),
    Effect.catchAll(() => Effect.succeed({ _tag: "Unavailable" as const, message: "Memory reading unavailable. Magnitude will try again shortly." })),
  )
  return ApplicationMemory.of({ read })
}))
export const nativeApplicationMemoryLayer = (addonPath: string) => applicationMemoryLayerFromLoader(() => createRequire(import.meta.url)(addonPath))

/** Subscription scope owns cadence; hidden windows don't trigger native work. */
export const observeApplicationMemory = (memory: ApplicationMemory, visible: () => boolean) => Stream.repeatEffectWithSchedule(
  Effect.suspend(() => visible() ? Effect.map(memory.read, Option.some) : Effect.succeed(Option.none<ApplicationMemoryObservation>())),
  Schedule.spaced("3 seconds"),
).pipe(Stream.filterMap(value => value))
