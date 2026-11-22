import { Deferred, Effect, Option, Ref } from "effect"
import { describe, expect, it } from "vitest"
import { ApplicationOwnershipFailed, HeadlessApplicationFailed, type HeadlessStop, type ServingBlocker } from "@magnitudedev/daemon-management/desktop-native"
import { runServeLoop } from "./serve-turn"

const ownershipError = new ApplicationOwnershipFailed({
  message: "Magnitude is already running. Stop the existing application or server before retrying.",
})
const headlessRunningError = new ApplicationOwnershipFailed({
  message: "Another `magnitude serve` command is already running. Stop it before starting another.",
})

const fastDelays = {
  blockerPollDelay: "1 millis" as const,
  desktopTakeoverPollDelay: "1 millis" as const,
  desktopTakeoverTimeoutMs: 50,
  ownershipReleasePollDelay: "1 millis" as const,
  ownershipReleaseMaxPolls: 5,
  portReleasePollDelay: "1 millis" as const,
  portReleaseMaxPolls: 5,
}

describe("serve turn-taking and takeover retry loop", () => {
  it("waits for an exiting desktop app to release application.lock after closing its control socket and port", async () => {
    const outcome = await Effect.runPromise(Effect.gen(function* () {
      const stopped = yield* Deferred.make<void>()
      const notices: string[] = []
      let desktopChecks = 0
      let lockPolls = 0
      let attempts = 0

      yield* runServeLoop({
        ...fastDelays,
        stopped,
        // Desktop is initially serving, then closes its control socket & port while still holding application.lock.
        blocker: Effect.sync(() => {
          desktopChecks++
          return desktopChecks <= 2 ? Option.some<ServingBlocker>("Desktop") : Option.none()
        }),
        observeOwner: Effect.succeed(Option.none()),
        ownershipHeld: Effect.sync(() => {
          lockPolls++
          return lockPolls < 3
        }),
        portBound: Effect.succeed(false),
        portLingering: Effect.succeed(false),
        notify: line => Effect.sync(() => { notices.push(line) }),
        onDesktopBlocker: Effect.void,
        replacedByDesktop: Effect.succeed(false),
        onReplacedByDesktop: Effect.void,
        onBeforeAttempt: Effect.void,
        runAttempt: Effect.suspend(() => {
          attempts++
          if (attempts === 1) return Effect.succeed<HeadlessStop>("DesktopTakeover")
          if (attempts === 2) return Effect.fail(ownershipError)
          return Effect.succeed<HeadlessStop>("Requested")
        }),
        onInstallUpdate: Effect.void,
      })

      return { attempts, lockPolls, notices }
    }))

    expect(outcome.attempts).toBe(3)
    expect(outcome.lockPolls).toBe(3)
    expect(outcome.notices).toContain("Another Magnitude app is serving on this machine; this server will resume when it stops.")
  })

  it("retries once when application.lock releases between admission failure and the retry check", async () => {
    let attempts = 0
    let blockerChecks = 0
    await Effect.runPromise(Effect.gen(function* () {
      const stopped = yield* Deferred.make<void>()
      yield* runServeLoop({
        ...fastDelays,
        stopped,
        blocker: Effect.sync(() => (++blockerChecks === 1 ? Option.some<ServingBlocker>("Desktop") : Option.none())),
        observeOwner: Effect.succeed(Option.none()),
        ownershipHeld: Effect.succeed(false),
        portBound: Effect.succeed(false),
        portLingering: Effect.succeed(false),
        notify: () => Effect.void,
        onDesktopBlocker: Effect.void,
        replacedByDesktop: Effect.succeed(false),
        onReplacedByDesktop: Effect.void,
        onBeforeAttempt: Effect.void,
        runAttempt: Effect.suspend(() => {
          attempts++
          return attempts === 1 ? Effect.fail(ownershipError) : Effect.succeed<HeadlessStop>("Requested")
        }),
        onInstallUpdate: Effect.void,
      })
    }))

    expect(attempts).toBe(2)
  })

  it("fails immediately when two concurrent magnitude serve processes contend without a prior wait", async () => {
    let attempts = 0
    let lockProbes = 0
    const result = await Effect.runPromise(Effect.gen(function* () {
      const stopped = yield* Deferred.make<void>()
      return yield* Effect.either(runServeLoop({
        ...fastDelays,
        stopped,
        blocker: Effect.succeed(Option.none()),
        observeOwner: Effect.succeed(Option.none()),
        ownershipHeld: Effect.sync(() => { lockProbes++; return true }),
        portBound: Effect.succeed(false),
        portLingering: Effect.succeed(false),
        notify: () => Effect.void,
        onDesktopBlocker: Effect.void,
        replacedByDesktop: Effect.succeed(false),
        onReplacedByDesktop: Effect.void,
        onBeforeAttempt: Effect.void,
        runAttempt: Effect.suspend(() => { attempts++; return Effect.fail(ownershipError) }),
        onInstallUpdate: Effect.void,
      }))
    }))

    expect(result._tag).toBe("Left")
    if (result._tag === "Left") expect(result.left).toEqual(ownershipError)
    expect(attempts).toBe(1)
    expect(lockProbes).toBe(0)
  })

  it("fails immediately without waiting when another headless serve owns the folder after a yield", async () => {
    let attempts = 0
    let lockProbes = 0
    let blockerChecks = 0
    const result = await Effect.runPromise(Effect.gen(function* () {
      const stopped = yield* Deferred.make<void>()
      return yield* Effect.either(runServeLoop({
        ...fastDelays,
        stopped,
        blocker: Effect.sync(() => (++blockerChecks === 1 ? Option.some<ServingBlocker>("Desktop") : Option.none())),
        observeOwner: Effect.succeed(Option.some("Headless" as const)),
        ownershipHeld: Effect.sync(() => { lockProbes++; return true }),
        portBound: Effect.succeed(false),
        portLingering: Effect.succeed(false),
        notify: () => Effect.void,
        onDesktopBlocker: Effect.void,
        replacedByDesktop: Effect.succeed(false),
        onReplacedByDesktop: Effect.void,
        onBeforeAttempt: Effect.void,
        runAttempt: Effect.suspend(() => { attempts++; return Effect.fail(headlessRunningError) }),
        onInstallUpdate: Effect.void,
      }))
    }))

    expect(result._tag).toBe("Left")
    if (result._tag === "Left") expect(result.left).toEqual(headlessRunningError)
    expect(attempts).toBe(1)
    expect(lockProbes).toBe(0)
  })

  it("fails with the ownership error if application.lock never releases within the retry budget", async () => {
    let attempts = 0
    let lockPolls = 0
    let blockerChecks = 0
    const result = await Effect.runPromise(Effect.gen(function* () {
      const stopped = yield* Deferred.make<void>()
      return yield* Effect.either(runServeLoop({
        ...fastDelays,
        stopped,
        blocker: Effect.sync(() => (++blockerChecks === 1 ? Option.some<ServingBlocker>("Desktop") : Option.none())),
        observeOwner: Effect.succeed(Option.none()),
        ownershipHeld: Effect.sync(() => { lockPolls++; return true }),
        portBound: Effect.succeed(false),
        portLingering: Effect.succeed(false),
        notify: () => Effect.void,
        onDesktopBlocker: Effect.void,
        replacedByDesktop: Effect.succeed(false),
        onReplacedByDesktop: Effect.void,
        onBeforeAttempt: Effect.void,
        runAttempt: Effect.suspend(() => { attempts++; return Effect.fail(ownershipError) }),
        onInstallUpdate: Effect.void,
      }))
    }))

    expect(result._tag).toBe("Left")
    if (result._tag === "Left") expect(result.left).toEqual(ownershipError)
    expect(attempts).toBe(2)
    expect(lockPolls).toBe(fastDelays.ownershipReleaseMaxPolls)
  })

  it("exits cleanly when a stop signal arrives while waiting for application.lock to release", async () => {
    let attempts = 0
    let blockerChecks = 0
    let lockPolls = 0
    const result = await Effect.runPromise(Effect.gen(function* () {
      const stopped = yield* Deferred.make<void>()
      return yield* Effect.either(runServeLoop({
        ...fastDelays,
        ownershipReleaseMaxPolls: 50,
        stopped,
        blocker: Effect.sync(() => (++blockerChecks === 1 ? Option.some<ServingBlocker>("Desktop") : Option.none())),
        observeOwner: Effect.succeed(Option.none()),
        ownershipHeld: Effect.gen(function* () {
          lockPolls++
          if (lockPolls === 2) yield* Deferred.succeed(stopped, undefined)
          return true
        }),
        portBound: Effect.succeed(false),
        portLingering: Effect.succeed(false),
        notify: () => Effect.void,
        onDesktopBlocker: Effect.void,
        replacedByDesktop: Effect.succeed(false),
        onReplacedByDesktop: Effect.void,
        onBeforeAttempt: Effect.void,
        runAttempt: Effect.suspend(() => { attempts++; return Effect.fail(ownershipError) }),
        onInstallUpdate: Effect.void,
      }))
    }))

    expect(result._tag).toBe("Right")
    expect(attempts).toBe(1)
    expect(lockPolls).toBe(2)
  })

  it("retries both ownership-lock release and subsequent port release independently", async () => {
    let attempts = 0
    let blockerChecks = 0
    let lockPolls = 0
    let portPolls = 0
    const portError = new HeadlessApplicationFailed({ message: "Port 10100 is already in use." })

    await Effect.runPromise(Effect.gen(function* () {
      const portBusy = yield* Ref.make(true)
      const stopped = yield* Deferred.make<void>()
      yield* runServeLoop({
        ...fastDelays,
        stopped,
        blocker: Effect.sync(() => (++blockerChecks === 1 ? Option.some<ServingBlocker>("Desktop") : Option.none())),
        observeOwner: Effect.succeed(Option.none()),
        ownershipHeld: Effect.sync(() => (++lockPolls < 2)),
        portBound: Ref.get(portBusy).pipe(Effect.tap(() => Effect.sync(() => {
          portPolls++
        })), Effect.flatMap(busy => {
          if (busy && portPolls >= 2) return Ref.set(portBusy, false).pipe(Effect.as(false))
          return Effect.succeed(busy)
        })),
        portLingering: Effect.succeed(false),
        notify: () => Effect.void,
        onDesktopBlocker: Effect.void,
        replacedByDesktop: Effect.succeed(false),
        onReplacedByDesktop: Effect.void,
        onBeforeAttempt: Effect.void,
        runAttempt: Effect.suspend(() => {
          attempts++
          if (attempts === 1) return Effect.fail<ApplicationOwnershipFailed | HeadlessApplicationFailed>(ownershipError)
          if (attempts === 2) return Effect.fail<ApplicationOwnershipFailed | HeadlessApplicationFailed>(portError)
          return Effect.succeed<HeadlessStop>("Requested")
        }),
        onInstallUpdate: Effect.void,
      })
    }))

    expect(attempts).toBe(3)
    expect(lockPolls).toBe(2)
  })
})
