import { Duration, Effect, Fiber, Option, TestClock, TestContext } from "effect"
import { describe, expect, it } from "vitest"
import type { PreparedUpdate, PreparedUpdateStore, UnattendedHold } from "@magnitudedev/daemon-management/desktop-native"
import type { UpdateOutcome } from "@magnitudedev/release/hosted-update"
import { CLI_VERSION } from "../version"
import { IdleInstallationSystem, makeDownloadedNotice, makeIdleInstallation } from "./serve-updates"

const release = { version: "99.0.0", bytes: 1, sha256: "a".repeat(64), signature: "A".repeat(86) + "==" }

const harness = (options: {
  readonly authorized?: boolean
  readonly idle?: readonly boolean[]
  readonly lockFree?: readonly boolean[]
  readonly installer?: "succeeds" | "fails" | "admission-fails"
  readonly outcome?: UpdateOutcome
  readonly hold?: UnattendedHold
} = {}) => {
  const events: string[] = []
  const notices: string[] = []
  let outcome = Option.fromNullable(options.outcome)
  let hold = Option.fromNullable(options.hold)
  let pending = Option.some<PreparedUpdate>({ release, from: Option.some("0.0.9"), installation: { _tag: "Unattempted" } })
  const idle = [...(options.idle ?? [true])]
  const lockFree = [...(options.lockFree ?? [true])]
  const store = {
    read: Effect.sync(() => pending),
    outcome: Effect.sync(() => outcome),
    recordOutcome: (value: UpdateOutcome) => Effect.sync(() => { outcome = Option.some(value); events.push(`outcome ${value.outcome}${Option.match(value.reason, { onNone: () => "", onSome: reason => `/${reason}` })}`) }),
    complete: () => Effect.sync(() => { outcome = Option.some({ outcome: "applied", version: release.version, from: Option.some("0.0.9"), reason: Option.none() }); pending = Option.none(); events.push("outcome applied") }),
    recordFailure: (_: unknown, kind: string) => Effect.sync(() => { events.push(`failed/${kind}`) }),
    discard: Effect.sync(() => { pending = Option.none(); events.push("discard") }),
    hold: Effect.sync(() => hold),
    recordHold: (value: UnattendedHold) => Effect.sync(() => { hold = Option.some(value); events.push(`hold ${Option.getOrElse(value.version, () => "-")} ${value.until}`) }),
  } as unknown as PreparedUpdateStore
  const system = IdleInstallationSystem.of({
    canInstallUnattended: Effect.sync(() => options.authorized ?? true),
    isIdle: Effect.sync(() => { const value = idle.length > 1 ? idle.shift()! : idle[0]!; events.push(value ? "idle" : "busy"); return value }),
    installationLockFree: Effect.sync(() => lockFree.length > 1 ? lockFree.shift()! : lockFree[0]!),
    store: Effect.succeed(store),
    installPrepared: options.installer === "admission-fails"
      ? Effect.sync(() => { events.push("install") }).pipe(Effect.zipRight(Effect.fail({ message: "Another Magnitude update is starting." })))
      : options.installer === "fails"
      ? Effect.sync(() => {
        events.push("install", "failed/install")
        pending = Option.map(pending, value => ({ ...value, installation: { _tag: "Failed", kind: "install", reason: "Package refused" } }) as PreparedUpdate)
      }).pipe(Effect.zipRight(Effect.fail({ message: "The package manager could not install the update." })))
      : Effect.gen(function* () { events.push("install"); yield* store.complete(release); events.push("exec") }),
    notify: line => Effect.sync(() => { notices.push(line) }),
  })
  const run = <A, E>(effect: (machine: Effect.Effect.Success<typeof makeIdleInstallation>) => Effect.Effect<A, E>) =>
    Effect.runPromise(makeIdleInstallation.pipe(Effect.flatMap(effect), Effect.provideService(IdleInstallationSystem, system), Effect.provide(TestContext.TestContext)))
  return { events, notices, run, outcome: () => outcome, hold: () => hold, pending: () => pending,
    markReported: Effect.sync(() => { outcome = Option.none() }) }
}

describe("install when idle", () => {
  it("installs at an idle point and records applied before replacing the process", async () => {
    const h = harness()
    await h.run(machine => Effect.gen(function* () {
      expect(yield* machine.installWhenIdle(release.version)).toBe(true)
      yield* machine.install
    }))
    expect(h.events).toEqual(["idle", "install", "outcome applied", "exec"])
  })
  it("waits while a turn or an inference request is active", async () => {
    const h = harness({ idle: [false, false, true] })
    await h.run(machine => Effect.gen(function* () {
      const decision = yield* machine.installWhenIdle(release.version).pipe(Effect.fork)
      yield* TestClock.adjust(Duration.seconds(10))
      expect(h.events).toEqual(["busy", "busy"])
      yield* TestClock.adjust(Duration.seconds(10))
      expect(yield* Fiber.join(decision)).toBe(true)
    }))
    expect(h.events).toEqual(["busy", "busy", "idle"])
  })
  it("keeps serving with one notice and no outcome when it cannot install without a password", async () => {
    const h = harness({ authorized: false })
    await h.run(machine => Effect.gen(function* () {
      expect(yield* machine.installWhenIdle(release.version)).toBe(false)
      expect(yield* machine.installWhenIdle(release.version)).toBe(false)
    }))
    expect(h.notices).toEqual([`Magnitude ${release.version} is downloaded. Install it with \`magnitude update install\`.`])
    expect(h.events).toEqual([])
    expect(Option.isNone(h.outcome())).toBe(true)
  })
  it("defers without stopping while another Magnitude holds the installation lock, reports it, then installs at a later idle point", async () => {
    const h = harness({ lockFree: [false, true] })
    const report = Effect.sync(() => { h.events.push("report") })
    await h.run(machine => Effect.gen(function* () {
      const decision = yield* machine.installWhenIdle(release.version, report).pipe(Effect.fork)
      yield* TestClock.adjust(Duration.minutes(14))
      // Still serving: no decision to stop, the deferral recorded and reported once.
      expect(h.events).toEqual(["idle", "outcome deferred", "hold - 900000", "report"])
      expect(h.notices).toEqual([`Magnitude ${release.version} is waiting for another Magnitude to quit; trying again later.`])
      yield* TestClock.adjust(Duration.minutes(1))
      expect(yield* Fiber.join(decision)).toBe(true)
      yield* machine.install
    }))
    expect(h.events).toEqual(["idle", "outcome deferred", "hold - 900000", "report", "idle", "install", "outcome applied", "exec"])
  })
  it("defers when the lock is taken between the idle decision and the installation", async () => {
    const h = harness({ lockFree: [true, false, true] })
    await h.run(machine => Effect.gen(function* () {
      expect(yield* machine.installWhenIdle(release.version)).toBe(true)
      yield* machine.install
      expect(h.events).toEqual(["idle", "outcome deferred", "hold - 900000"])
      expect(h.outcome()).toEqual(Option.some({ outcome: "deferred", version: release.version, from: Option.some("0.0.9"), reason: Option.none() }))
      const retry = yield* machine.installWhenIdle(release.version).pipe(Effect.fork)
      yield* TestClock.adjust(Duration.minutes(15))
      expect(yield* Fiber.join(retry)).toBe(true)
    }))
    expect(h.events).toEqual(["idle", "outcome deferred", "hold - 900000", "idle"])
  })
  it("returns to serving the current version when the package fails, holding that version back", async () => {
    const h = harness({ installer: "fails" })
    await h.run(machine => machine.install)
    // The failed download is dropped so a newer release can be fetched; its outcome stays to be reported.
    expect(h.events).toEqual(["install", "failed/install", `hold ${release.version} 3600000`, "outcome failed/install", "discard"])
    expect(h.outcome()).toEqual(Option.some({ outcome: "failed", version: release.version, from: Option.some("0.0.9"), reason: Option.some("install") }))
    expect(h.notices.at(-1)).toContain("Serving the current version.")
  })
  it("waits out the hold after a failed install, even in a restarted owner, instead of trying again at once", async () => {
    const h = harness({ installer: "admission-fails" })
    await h.run(machine => machine.install)
    // Admission failed before the package ran: the download stays, with no version held back and no outcome.
    expect(h.events).toEqual(["install", "hold - 3600000"])
    expect(Option.isSome(h.pending())).toBe(true)
    expect(Option.isNone(h.outcome())).toBe(true)
    await h.run(machine => Effect.gen(function* () {
      const decision = yield* machine.installWhenIdle(release.version).pipe(Effect.fork)
      yield* TestClock.adjust(Duration.minutes(59))
      expect(h.events).toEqual(["install", "hold - 3600000"])
      yield* TestClock.adjust(Duration.minutes(1))
      expect(yield* Fiber.join(decision)).toBe(true)
    }))
    expect(h.events).toEqual(["install", "hold - 3600000", "idle"])
  })
  it("keeps a held-back version when a later attempt fails without running the package", async () => {
    const h = harness({ installer: "admission-fails", hold: { version: Option.some("98.0.0"), until: 0 } })
    await h.run(machine => machine.install)
    expect(h.hold()).toEqual(Option.some({ version: Option.some("98.0.0"), until: 3_600_000 }))
  })
  it("turns an unreported applied outcome into failed/startup and reports it before exiting", async () => {
    const h = harness({ outcome: { outcome: "applied", version: CLI_VERSION, from: Option.some("0.0.9"), reason: Option.none() } })
    const reports: Array<Option.Option<UpdateOutcome>> = []
    const report = Effect.sync(() => { reports.push(h.outcome()) })
    await h.run(machine => machine.startupFailed(report))
    expect(reports).toEqual([Option.some({ outcome: "failed", version: CLI_VERSION, from: Option.some("0.0.9"), reason: Option.some("startup") })])
  })
  it("reports a startup failure once: restarts after it was sent make no request", async () => {
    const h = harness({ outcome: { outcome: "applied", version: CLI_VERSION, from: Option.some("0.0.9"), reason: Option.none() } })
    let reports = 0
    const report = Effect.sync(() => { reports += 1 }).pipe(Effect.zipRight(h.markReported))
    for (let restart = 0; restart < 3; restart++) await h.run(machine => machine.startupFailed(report))
    expect(reports).toBe(1)
  })
  it("leaves other outcomes alone when a start fails", async () => {
    const h = harness({ outcome: { outcome: "applied", version: "0.0.1", from: Option.some("0.0.0"), reason: Option.none() } })
    await h.run(machine => machine.startupFailed(Effect.void))
    expect(h.events).toEqual([])
  })
})

describe("downloaded notice", () => {
  it("prints one line per downloaded version, as Windows `serve` does for each update it prepares", async () => {
    const lines: string[] = []
    await Effect.runPromise(Effect.gen(function* () {
      const notice = yield* makeDownloadedNotice(line => Effect.sync(() => { lines.push(line) }))
      yield* notice("2.0.0")
      yield* notice("2.0.0")
      yield* notice("2.1.0")
    }))
    expect(lines).toEqual([
      "Magnitude 2.0.0 is downloaded. Install it with `magnitude update install`.",
      "Magnitude 2.1.0 is downloaded. Install it with `magnitude update install`.",
    ])
  })
})
