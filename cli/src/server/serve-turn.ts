import { Deferred, Duration, Effect, Option, Ref } from "effect"
import type { HeadlessStop, ServingBlocker } from "@magnitudedev/daemon-management/desktop-native"

export interface ServeLoopOptions<E, R = never> {
  readonly stopped: Deferred.Deferred<void>
  readonly blocker: Effect.Effect<Option.Option<ServingBlocker>, E, R>
  readonly observeOwner: Effect.Effect<Option.Option<"Desktop" | "Headless">, E, R>
  readonly ownershipHeld: Effect.Effect<boolean, E, R>
  readonly portBound: Effect.Effect<boolean, E, R>
  readonly portLingering: Effect.Effect<boolean, E, R>
  readonly notify: (line: string) => Effect.Effect<void, E, R>
  readonly onDesktopBlocker: Effect.Effect<void, E, R>
  readonly replacedByDesktop: Effect.Effect<boolean, E, R>
  readonly onReplacedByDesktop: Effect.Effect<void, E, R>
  readonly onBeforeAttempt: Effect.Effect<void, E, R>
  readonly runAttempt: Effect.Effect<HeadlessStop, E, R>
  readonly onInstallUpdate: Effect.Effect<void, E, R>
  readonly blockerPollDelay?: Duration.DurationInput
  readonly desktopTakeoverPollDelay?: Duration.DurationInput
  readonly desktopTakeoverTimeoutMs?: number
  readonly ownershipReleasePollDelay?: Duration.DurationInput
  readonly ownershipReleaseMaxPolls?: number
  readonly portReleasePollDelay?: Duration.DurationInput
  readonly portReleaseMaxPolls?: number
}

const isOwnershipContentionError = (error: unknown): boolean =>
  typeof error === "object" && error !== null && "_tag" in error && error._tag === "ApplicationOwnershipFailed"

/**
 * Coordinates `magnitude serve`'s turn-taking and restart loop: waits while the desktop app, an update installation,
 * or another Magnitude serves; retries cleanly when an exiting owner is still releasing `application.lock` or its
 * TCP port; and still fails immediately when two `magnitude serve` commands contend directly.
 */
export const runServeLoop = <E, R = never>(options: ServeLoopOptions<E, R>): Effect.Effect<void, E, R> => Effect.gen(function* () {
  const {
    stopped,
    blocker,
    observeOwner,
    ownershipHeld,
    portBound,
    portLingering,
    notify,
    onDesktopBlocker,
    replacedByDesktop,
    onReplacedByDesktop,
    onBeforeAttempt,
    runAttempt,
    onInstallUpdate,
    blockerPollDelay = "2 seconds",
    desktopTakeoverPollDelay = "250 millis",
    desktopTakeoverTimeoutMs = 60_000,
    ownershipReleasePollDelay = "250 millis",
    ownershipReleaseMaxPolls = 60,
    portReleasePollDelay = "1 second",
    portReleaseMaxPolls = 90,
  } = options

  const waited = yield* Ref.make(false)
  const retriedOwnership = yield* Ref.make(false)
  const retriedPort = yield* Ref.make(false)

  /** Waits while the desktop app or another Magnitude serves, or an update installs; resolves once this server may start. */
  const awaitTurn = Effect.gen(function* () {
    let announced = false
    for (let current = yield* blocker; Option.isSome(current); current = yield* blocker) {
      if (!announced) yield* notify(current.value === "UpdateInstallation"
        ? "A Magnitude update is being installed; this server will resume when it finishes."
        : "Another Magnitude app is serving on this machine; this server will resume when it stops.")
      announced = true
      yield* Ref.set(waited, true)
      if (yield* Deferred.isDone(stopped)) return false
      if (current.value === "Desktop") yield* onDesktopBlocker
      yield* Effect.raceFirst(Effect.sleep(blockerPollDelay), Deferred.await(stopped))
    }
    return !(yield* Deferred.isDone(stopped))
  })

  /**
   * After yielding, the desktop still has to take the ownership lock (or, from another account, the port) this server
   * just released; until it has, nothing appears to serve. Wait for it to (or a minute, if it never does) so this
   * server does not take them straight back.
   */
  const awaitDesktop = Effect.gen(function* () {
    const deadline = Date.now() + desktopTakeoverTimeoutMs
    while (Date.now() < deadline && !(yield* Deferred.isDone(stopped))) {
      if (Option.isSome(yield* blocker)) return
      yield* Effect.raceFirst(Effect.sleep(desktopTakeoverPollDelay), Deferred.await(stopped))
    }
  })

  // A restart request re-admits the owner in place, so settings read at service start take effect.
  for (;;) {
    yield* onBeforeAttempt
    if (!(yield* awaitTurn)) return
    if (yield* replacedByDesktop) {
      yield* onReplacedByDesktop
      return
    }
    const attempt = yield* Effect.either(runAttempt)
    // Another Magnitude may have started between the wait and admission; wait for it rather than fail.
    if (attempt._tag === "Left") {
      if (Option.isSome(yield* blocker)) continue
      // An exiting desktop app stops its service and closes its control socket before releasing `application.lock`.
      // If we just waited on or yielded to another owner and no headless `serve` answers, wait briefly for the lock
      // to clear and retry once (also covering the race where the lock cleared right after admission failed).
      if (isOwnershipContentionError(attempt.left) && (yield* Ref.get(waited)) && !(yield* Ref.get(retriedOwnership))) {
        const owner = yield* observeOwner
        if (!Option.contains(owner, "Headless")) {
          let headlessObserved = false
          for (let poll = 0; poll < ownershipReleaseMaxPolls; poll++) {
            if (yield* Deferred.isDone(stopped)) return
            if (Option.isSome(yield* blocker)) break
            const currentOwner = yield* observeOwner
            if (Option.contains(currentOwner, "Headless")) {
              headlessObserved = true
              break
            }
            if (!(yield* ownershipHeld)) break
            yield* Effect.raceFirst(Effect.sleep(ownershipReleasePollDelay), Deferred.await(stopped))
          }
          if (yield* Deferred.isDone(stopped)) return
          if (!headlessObserved) {
            yield* Ref.set(retriedOwnership, true)
            continue
          }
        }
      }
      // A Magnitude that just stopped still holds the port for a while: it stops answering before it closes its
      // listener, and its closed connections linger after (up to 30 s on macOS, where they keep another account
      // from binding). Wait for the port while it is being released, then try again once.
      const releasing = Effect.zipWith(Ref.get(waited), portBound, (didWait, bound) => didWait && bound).pipe(
        Effect.flatMap(isReleasing => isReleasing ? Effect.succeed(true) : portLingering),
      )
      if (!(yield* Ref.get(retriedPort)) && (yield* releasing)) {
        for (let poll = 0; poll < portReleaseMaxPolls && (yield* portBound) && !(yield* Deferred.isDone(stopped)); poll++) {
          yield* Effect.raceFirst(Effect.sleep(portReleasePollDelay), Deferred.await(stopped))
        }
        if (yield* Deferred.isDone(stopped)) return
        yield* Ref.set(waited, false)
        yield* Ref.set(retriedPort, true)
        continue
      }
      return yield* Effect.fail(attempt.left)
    }
    const stop = attempt.right
    yield* Ref.set(waited, false)
    yield* Ref.set(retriedOwnership, false)
    yield* Ref.set(retriedPort, false)
    // Installation replaces this process on success; otherwise the current version serves again.
    if (stop === "InstallUpdate") yield* onInstallUpdate
    else if (stop === "DesktopTakeover") {
      yield* Ref.set(waited, true)
      yield* awaitDesktop
    } else if (stop !== "Restart") return
  }
})
