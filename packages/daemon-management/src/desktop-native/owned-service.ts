import { Starting, Ready, Failed, CleanupFailed, Stopping, Stopped, OwnedServiceState } from "@magnitudedev/sdk/desktop-host"
import { type MagnitudeHealthResponse } from "@magnitudedev/acn-protocol"
import { FSM } from "@magnitudedev/utils"
import { Cause, Clock, Deferred, Effect, Exit, Fiber, Option, Queue, Ref, Schema, Stream, SubscriptionRef } from "effect"
import { OwnedChildSpawner, type OwnedChild, type OwnedChildCommand } from "./owned-child"

const states = { Starting, Ready, Failed, CleanupFailed, Stopping, Stopped }
const machine = FSM.defineFSM(states, {
  Starting: ["Ready", "Failed", "CleanupFailed", "Stopping"],
  Ready: ["Starting", "Failed", "CleanupFailed", "Stopping"],
  Failed: ["Starting", "Stopping", "CleanupFailed"],
  CleanupFailed: ["Stopping"],
  Stopping: ["Stopped", "CleanupFailed"],
  Stopped: [],
} as const)
export { OwnedServiceState } from "@magnitudedev/sdk/desktop-host"

export class ServiceChildExited extends Schema.TaggedError<ServiceChildExited>()("ServiceChildExited", {
  code: Schema.Number, diagnostic: Schema.String,
}) { override get message() { return `Magnitude service exited unexpectedly (exit code ${this.code}).` } }
export class ServiceChildProtocolFailed extends Schema.TaggedError<ServiceChildProtocolFailed>()("ServiceChildProtocolFailed", {
  message: Schema.String,
}) {}
export class ServiceStartupTimedOut extends Schema.TaggedError<ServiceStartupTimedOut>()("ServiceStartupTimedOut", {}) {
  override get message() { return "Service startup timed out" }
}
export class OwnedServiceUnavailable extends Schema.TaggedError<OwnedServiceUnavailable>()("OwnedServiceUnavailable", {
  message: Schema.String,
}) {}

/** One application-scoped supervisor; observers and renderer lifetimes do not own it. */
export const makeOwnedService = (command: OwnedChildCommand, rpcVersion: number) => Effect.gen(function* () {
  const spawner = yield* OwnedChildSpawner
  const status = yield* SubscriptionRef.make<OwnedServiceState>(new Starting({ attempt: 0, health: Option.none() }))
  const active = yield* Ref.make(Option.none<OwnedChild>())
  const cleanupFailure = yield* Ref.make(Option.none<string>())
  const retry = yield* Queue.sliding<void>(1)
  const transition = yield* Effect.makeSemaphore(1)
  const quitLock = yield* Effect.makeSemaphore(1)
  const change = (f: (state: OwnedServiceState) => OwnedServiceState) => transition.withPermits(1)(SubscriptionRef.update(status, f))
  const terminal = (state: OwnedServiceState) => state._tag === "Stopping" || state._tag === "Stopped" || state._tag === "CleanupFailed"

  const retire = (child: OwnedChild) => child.stop.pipe(
    Effect.tap(() => Ref.set(active, Option.none())),
    Effect.catchAll(error => Ref.set(cleanupFailure, Option.some(error.message))),
  )
  const attempt = (number: number, readyAt: Ref.Ref<Option.Option<number>>) => Effect.scoped(Effect.gen(function* () {
    const child = yield* spawner.spawn(command)
    yield* Ref.set(active, Option.some(child))
    yield* Effect.addFinalizer(() => retire(child))
    const admitted = yield* Ref.make(false)
    const instance = yield* Ref.make(Option.none<string>())
    const ready = yield* Deferred.make<void>()
    const stoppingDetail = yield* Ref.make(Option.none<string>())
    const observe = child.events.pipe(Stream.runForEach(event => Effect.gen(function* () {
      if (event._tag === "Booted") {
        if (event.pid !== child.identity.pid || (yield* Ref.get(admitted))) {
          return yield* new ServiceChildProtocolFailed({ message: "Unexpected service bootstrap identity" })
        }
        yield* Ref.set(admitted, true)
        yield* child.send({ _tag: "Start" })
        return
      }
      const health = event.health
      const currentId = yield* Ref.get(instance)
      if (!(yield* Ref.get(admitted)) || health.pid !== child.identity.pid || health.rpcVersion !== rpcVersion ||
          (Option.isSome(currentId) && currentId.value !== health.id)) {
        return yield* new ServiceChildProtocolFailed({ message: "Service health does not match its owned process or RPC contract" })
      }
      yield* Ref.set(instance, Option.some(health.id))
      if (health.state._tag === "Ready") {
        if (!(yield* Deferred.isDone(ready))) yield* Ref.set(readyAt, Option.some(yield* Clock.currentTimeMillis))
        yield* change(state => terminal(state) ? state : state._tag === "Ready"
          ? machine.hold(state, { health }) : state._tag === "Starting"
            ? machine.transition(state, "Ready", { health }) : state)
        yield* Deferred.succeed(ready, undefined)
      } else if (health.state._tag === "Stopping") {
        const stopping = health.state
        const message = Option.getOrElse(stopping.safeDetail, () =>
          stopping.reason === "startup-failed" ? "Magnitude service could not start."
            : stopping.reason === "icn-exited" ? "The inference engine stopped unexpectedly."
            : "Magnitude service stopped unexpectedly.")
        // Retain the observed result before acknowledgement permits child exit.
        yield* Ref.set(stoppingDetail, Option.some(message))
        yield* child.send({ _tag: "StoppingObserved" }).pipe(Effect.ignore)
        return yield* new ServiceChildProtocolFailed({ message })
      } else if (health.state._tag === "Starting") {
        yield* change(state => state._tag === "Starting" ? machine.hold(state, { attempt: number, health: Option.some(health) }) : state)
      }
    })), Effect.zipRight(Effect.fail(new ServiceChildProtocolFailed({ message: "Service control stream ended" }))))
    const startupDeadline = Deferred.await(ready).pipe(
      Effect.timeoutFail({ duration: "5 minutes", onTimeout: () => new ServiceStartupTimedOut({}) }),
      Effect.zipRight(Effect.never),
    )
    return yield* Effect.raceFirst(observe, Effect.raceFirst(
      child.exit.pipe(Effect.flatMap(code => Effect.gen(function* () {
        const detail = yield* Ref.get(stoppingDetail)
        if (Option.isSome(detail)) return yield* new ServiceChildProtocolFailed({ message: detail.value })
        return yield* new ServiceChildExited({ code, diagnostic: yield* child.diagnosticTail })
      }))),
      startupDeadline,
    )).pipe(Effect.tapErrorCause(cause => Cause.isInterruptedOnly(cause) ? Effect.void :
      child.diagnosticTail.pipe(Effect.flatMap(diagnostic => diagnostic
        ? Effect.logError("Magnitude service diagnostics", diagnostic) : Effect.void))))
  }))

  const worker = yield* Effect.gen(function* () {
    let failures = 0
    for (;;) {
      const readyAt = yield* Ref.make(Option.none<number>())
      const result = yield* attempt(failures, readyAt).pipe(Effect.exit)
      if (Exit.isFailure(result) && Cause.isInterruptedOnly(result.cause)) return yield* Effect.interrupt
      const cleanup = yield* Ref.get(cleanupFailure)
      if (Option.isSome(cleanup)) {
        yield* change(state => terminal(state) ? state : machine.transition(state, "CleanupFailed", { message: cleanup.value }))
        return
      }
      const now = yield* Clock.currentTimeMillis
      const lastReady = yield* Ref.get(readyAt)
      if (Option.isSome(lastReady) && now - lastReady.value >= 30_000) failures = 0
      if (failures >= 3) {
        if (Exit.isFailure(result)) yield* Effect.logError(result.cause)
        const failure = Exit.isFailure(result) ? Cause.failureOption(result.cause) : Option.none()
        const message = Option.isSome(failure) && failure.value.message
          ? failure.value.message : "Magnitude service stopped unexpectedly."
        yield* change(state => terminal(state) ? state : machine.transition(state, "Failed", { message }))
        yield* Queue.take(retry)
        failures = 0
      } else {
        failures += 1
        yield* change(state => terminal(state) ? state : state._tag === "Starting"
          ? machine.hold(state, { attempt: failures, health: Option.none() })
          : machine.transition(state, "Starting", { attempt: failures, health: Option.none() }))
        yield* Effect.sleep(`${2 ** (failures - 1)} seconds`)
      }
      yield* change(state => terminal(state) ? state : state._tag === "Starting"
        ? machine.hold(state, { attempt: failures, health: Option.none() })
        : machine.transition(state, "Starting", { attempt: failures, health: Option.none() }))
    }
  }).pipe(Effect.forkScoped)

  const shutdown = quitLock.withPermits(1)(Effect.gen(function* () {
    const current = yield* SubscriptionRef.get(status)
    if (current._tag === "Stopped") return
    yield* change(state => state._tag === "Stopped" || state._tag === "Stopping" ? state : machine.transition(state, "Stopping", {}))
    const child = yield* Ref.get(active)
    // The worker is interrupted after the cooperative request so it cannot restart a stopping service.
    if (Option.isSome(child)) {
      yield* child.value.send({ _tag: "Shutdown" }).pipe(Effect.ignore)
    }
    yield* Fiber.interrupt(worker)
    const remaining = yield* Ref.get(active)
    if (Option.isSome(remaining)) {
      yield* remaining.value.stop.pipe(Effect.mapError(error => new OwnedServiceUnavailable({ message: error.message })))
      yield* Ref.set(active, Option.none())
    }
    yield* change(state => state._tag === "Stopping" ? machine.transition(state, "Stopped", {}) : state)
  }).pipe(
    Effect.tapError(error => change(state => state._tag === "Stopping" ? machine.transition(state, "CleanupFailed", { message: error.message }) : state)),
    Effect.uninterruptible,
  ))
  yield* Effect.addFinalizer(() => shutdown.pipe(Effect.catchAll(Effect.logError)))
  const awaitReady = status.changes.pipe(Stream.filter(state => state._tag !== "Starting"), Stream.take(1), Stream.runHead,
    Effect.flatMap(Option.match({
      onNone: () => Effect.fail(new OwnedServiceUnavailable({ message: "Application service observation ended" })),
      onSome: state => state._tag === "Ready" ? Effect.succeed(state.health) : Effect.fail(new OwnedServiceUnavailable({
        message: state._tag === "Failed" || state._tag === "CleanupFailed" ? state.message : "Application is stopping",
      })),
    })))
  return {
    state: SubscriptionRef.get(status), changes: status.changes, awaitReady, shutdown,
    retry: transition.withPermits(1)(Effect.gen(function* () {
      if ((yield* SubscriptionRef.get(status))._tag === "Failed") yield* Queue.offer(retry, undefined)
    })),
  }
})
