/**
 * File watch subscription — managed WatchFile fiber.
 *
 * WatchFile is a resident stream (per SDK spec)
 * with the same liveness/reconnection handling as StreamDisplayView.
 * The recoveringProtocolLayer handles transport recovery.
 *
 * Module-level singleton — one file watch at a time (the file panel watches
 * one file). Interrupting the previous fiber before starting a new one.
 */
import { Effect, Stream, Layer, Fiber, Cause } from "effect"
import { RpcClient } from "@effect/rpc"
import { FetchHttpClient } from "@effect/platform"
import {
  MagnitudeRpcs,
  recoveringProtocolLayer,
  type DaemonSpawnerTag,
  type WatchFileWireEvent,
  type WatchFileEvent,
} from "@magnitudedev/sdk"

/** Callbacks for file watch lifecycle events */
export interface FileWatchCallbacks {
  /** Called when a file event (created/changed/removed) is received */
  readonly onEvent: (event: WatchFileEvent) => void
  /** Called when the watch stream fails with a non-interrupt error */
  readonly onError: (message: string) => void
}

let currentFiber: Fiber.RuntimeFiber<void, unknown> | null = null

const buildStreamLayer = (daemonSpawnerLayer: Layer.Layer<DaemonSpawnerTag, never, never>) =>
  recoveringProtocolLayer().pipe(
    Layer.provide(Layer.mergeAll(FetchHttpClient.layer, daemonSpawnerLayer)),
  )

const isFileEvent = (event: WatchFileWireEvent): event is WatchFileEvent =>
  !("_tag" in event)

/**
 * Subscribe to WatchFile for a session + path.
 * Interrupts the previous fiber if one is active.
 */
export function subscribeFileWatch(
  daemonSpawnerLayer: Layer.Layer<DaemonSpawnerTag, never, never>,
  cwd: string,
  path: string,
  callbacks: FileWatchCallbacks,
): void {
  interruptFileWatch()

  const watchEffect = Effect.gen(function* () {
    const client = yield* RpcClient.make(MagnitudeRpcs)
    const stream = client.WatchFile({ cwd, path }).pipe(
      Stream.filter(isFileEvent),
      Stream.tap((event) =>
        Effect.sync(() => callbacks.onEvent(event)),
      ),
      Stream.runDrain,
    )
    yield* stream
  }).pipe(
    Effect.catchAllCause((cause) =>
      Cause.isInterruptedOnly(cause)
        ? Effect.void
        : Effect.gen(function* () {
            const failure = Cause.failureOption(cause)
            const message = failure._tag === "Some" && failure.value instanceof Error
              ? failure.value.message
              : "File watch failed"
            callbacks.onError(message)
            yield* Effect.logError(`WatchFile error: ${cause}`)
          }),
    ),
    Effect.scoped,
    Effect.provide(buildStreamLayer(daemonSpawnerLayer)),
  )

  currentFiber = Effect.runFork(watchEffect)
}

/**
 * Interrupt the current file watch fiber immediately.
 */
export function interruptFileWatch(): void {
  if (currentFiber) {
    Effect.runFork(Fiber.interrupt(currentFiber))
    currentFiber = null
  }
}
