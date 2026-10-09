import { Duration, Effect, Fiber, Option, TestClock, TestContext } from "effect"
import { describe, expect, it } from "vitest"
import type { PreparedUpdate, PreparedUpdateStore } from "@magnitudedev/daemon-management/desktop-native"
import type { UpdateOutcome } from "@magnitudedev/release/hosted-update"
import { CLI_VERSION } from "../version"
import { IdleInstallationSystem, makeIdleInstallation } from "./serve-updates"

const release = { version: "99.0.0", bytes: 1, sha256: "a".repeat(64), signature: "A".repeat(86) + "==" }

const harness = (options: {
  readonly authorized?: boolean
  readonly idle?: readonly boolean[]
  readonly lockFree?: readonly boolean[]
  readonly installer?: "succeeds" | "fails"
  readonly outcome?: UpdateOutcome
} = {}) => {
  const events: string[] = []
  const notices: string[] = []
  let outcome = Option.fromNullable(options.outcome)
  let pending = Option.some<PreparedUpdate>({ release, installation: { _tag: "Unattempted" } })
  const idle = [...(options.idle ?? [true])]
  const lockFree = [...(options.lockFree ?? [true])]
  const store = {
    read: Effect.sync(() => pending),
    outcome: Effect.sync(() => outcome),
    recordOutcome: (value: UpdateOutcome) => Effect.sync(() => { outcome = Option.some(value); events.push(`outcome ${value.outcome}${Option.match(value.reason, { onNone: () => "", onSome: reason => `/${reason}` })}`) }),
    complete: () => Effect.sync(() => { outcome = Option.some({ outcome: "applied", version: release.version, reason: Option.none() }); pending = Option.none(); events.push("outcome applied") }),
    recordFailure: (_: unknown, kind: string) => Effect.sync(() => { events.push(`failed/${kind}`) }),
  } as unknown as PreparedUpdateStore
  const system = IdleInstallationSystem.of({
    canInstallUnattended: Effect.sync(() => options.authorized ?? true),
    isIdle: Effect.sync(() => { const value = idle.length > 1 ? idle.shift()! : idle[0]!; events.push(value ? "idle" : "busy"); return value }),
    installationLockFree: Effect.sync(() => lockFree.length > 1 ? lockFree.shift()! : lockFree[0]!),
    store: Effect.succeed(store),
    installPrepared: options.installer === "fails"
      ? Effect.sync(() => { events.push("install", "failed/install") }).pipe(Effect.zipRight(Effect.fail({ message: "The package manager could not install the update." })))
      : Effect.gen(function* () { events.push("install"); yield* store.complete(release); events.push("exec") }),
    notify: line => Effect.sync(() => { notices.push(line) }),
  })
  const run = <A, E>(effect: (machine: Effect.Effect.Success<typeof makeIdleInstallation>) => Effect.Effect<A, E>) =>
    Effect.runPromise(makeIdleInstallation.pipe(Effect.flatMap(effect), Effect.provideService(IdleInstallationSystem, system), Effect.provide(TestContext.TestContext)))
  return { events, notices, run, outcome: () => outcome }
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
  it("keeps serving with one notice and no outcome without unattended authorization", async () => {
    const h = harness({ authorized: false })
    await h.run(machine => Effect.gen(function* () {
      expect(yield* machine.installWhenIdle(release.version)).toBe(false)
      expect(yield* machine.installWhenIdle(release.version)).toBe(false)
    }))
    expect(h.notices).toEqual([`Magnitude ${release.version} is downloaded. Install it with \`magnitude update install\`.`])
    expect(h.events).toEqual([])
    expect(Option.isNone(h.outcome())).toBe(true)
  })
  it("defers while another Magnitude holds the installation lock, then installs at a later idle point", async () => {
    const h = harness({ lockFree: [false, true] })
    await h.run(machine => Effect.gen(function* () {
      yield* machine.install
      expect(h.events).toEqual(["outcome deferred"])
      const retry = yield* machine.installWhenIdle(release.version).pipe(Effect.fork)
      yield* TestClock.adjust(Duration.minutes(14))
      expect(h.events).toEqual(["outcome deferred"])
      yield* TestClock.adjust(Duration.minutes(1))
      expect(yield* Fiber.join(retry)).toBe(true)
      yield* machine.install
    }))
    expect(h.events).toEqual(["outcome deferred", "idle", "install", "outcome applied", "exec"])
  })
  it("returns to serving the current version when the package fails", async () => {
    const h = harness({ installer: "fails" })
    await h.run(machine => machine.install)
    expect(h.events).toEqual(["install", "failed/install"])
    expect(h.notices.at(-1)).toContain("Serving the current version.")
  })
  it("turns an unreported applied outcome into failed/startup when the new version does not start", async () => {
    const h = harness({ outcome: { outcome: "applied", version: CLI_VERSION, reason: Option.none() } })
    await h.run(machine => machine.startupFailed)
    expect(h.outcome()).toEqual(Option.some({ outcome: "failed", version: CLI_VERSION, reason: Option.some("startup") }))
  })
  it("leaves other outcomes alone when a start fails", async () => {
    const h = harness({ outcome: { outcome: "applied", version: "0.0.1", reason: Option.none() } })
    await h.run(machine => machine.startupFailed)
    expect(h.events).toEqual([])
  })
})
