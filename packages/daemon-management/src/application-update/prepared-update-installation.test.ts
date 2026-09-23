import { Effect, Option } from "effect"
import { describe, expect, it } from "vitest"
import { PreparedUpdateFailed, PreparedUpdateStore, type PreparedUpdate } from "@magnitudedev/daemon-management/desktop-native"
import { ApplicationUpdateFailed } from "./application-update"
import { PreparedUpdateInstaller, installPreparedUpdate, reconcilePreparedUpdate, preparedUpdateFailure } from "./prepared-update-installation"

const release = { version: "2.0.0", bytes: 1, sha256: "a".repeat(64), signature: "A".repeat(86) + "==" }
const harness = (installation: PreparedUpdate["installation"] = { _tag: "Unattempted" }) => {
  let pending = Option.some<PreparedUpdate>({ release, installation })
  const events: string[] = []
  const store: PreparedUpdateStore = {
    read: Effect.sync(() => pending),
    prepare: () => Effect.die("No download belongs in restart recovery"),
    verify: () => Effect.sync(() => { events.push("verify"); return "retained-installer" }),
    recordAttempt: () => Effect.sync(() => { events.push("attempt"); pending = Option.some({ release, installation: { _tag: "Attempted" } }) }),
    recordFailure: (_, reason) => Effect.sync(() => { events.push("failure"); pending = Option.some({ release, installation: { _tag: "Failed", reason } }) }),
    discard: Effect.sync(() => { events.push("discard"); pending = Option.none() }),
    removeAbandonedTransfers: Effect.sync(() => { events.push("cleanup"); }),
  }
  const installer: PreparedUpdateInstaller = { requiresAuthorization: false, install: archive => Effect.sync(() => {
    expect(archive).toBe("retained-installer")
    expect(Option.getOrThrow(pending).installation._tag).toBe("Attempted")
    events.push("install")
  }) }
  const run = <A, E>(effect: Effect.Effect<A, E, PreparedUpdateStore | PreparedUpdateInstaller>, overrides: Partial<PreparedUpdateStore> = {}, native: Partial<PreparedUpdateInstaller> = {}) =>
    Effect.runPromise(effect.pipe(Effect.provideService(PreparedUpdateStore, { ...store, ...overrides }), Effect.provideService(PreparedUpdateInstaller, { ...installer, ...native })))
  return { run, events, pending: () => pending }
}

describe("prepared update installation", () => {
  it("passes caller continuation through the verified durable attempt barrier", async () => {
    const h = harness()
    expect(await h.run(installPreparedUpdate({ continuation: { _tag: "Caller" }, allowAuthorizationPrompt: false }), {}, {
      install: (_archive, _release, continuation) => Effect.sync(() => {
        expect(continuation).toEqual({ _tag: "Caller" })
        expect(Option.getOrThrow(h.pending()).installation._tag).toBe("Attempted")
        h.events.push("install")
      }),
    })).toBe("Started")
    expect(h.events).toEqual(["verify", "attempt", "install"])
  })
  it.each(["Unattempted", "Attempted", "Failed"] as const)("cleans a completed %s update based on installed version", async tag => {
    const h = harness(tag === "Failed" ? { _tag: tag, reason: "previous failure" } : { _tag: tag })
    expect(Option.isNone(await h.run(reconcilePreparedUpdate("3.0.0")))).toBe(true)
    expect(h.events).toEqual(["cleanup", "discard"])
  })
  it("retains an interrupted attempt without inventing an installer error", async () => {
    const h = harness({ _tag: "Attempted" })
    const pending = Option.getOrThrow(await h.run(reconcilePreparedUpdate("1.0.0")))
    expect(preparedUpdateFailure(pending)).toEqual(Option.some("The update did not complete."))
    expect(h.events).toEqual(["cleanup"])
  })
  it.each(["Unattempted", "Attempted", "Failed"] as const)("uses the same retained bytes and attempt barrier for explicit %s installation", async tag => {
    const h = harness(tag === "Failed" ? { _tag: tag, reason: "cancelled" } : { _tag: tag })
    expect(await h.run(installPreparedUpdate({ continuation: { _tag: "Desktop", showWindow: true }, allowAuthorizationPrompt: true }))).toBe("Started")
    expect(h.events).toEqual(["verify", "attempt", "install"])
    expect(Option.getOrThrow(h.pending()).installation).toEqual({ _tag: "Attempted" })
  })
  it("defers a background launch before verification, attempting or showing authorization", async () => {
    const h = harness()
    expect(await h.run(installPreparedUpdate({ continuation: { _tag: "Desktop", showWindow: false }, allowAuthorizationPrompt: false }), {}, { requiresAuthorization: true })).toBe("Deferred")
    expect(h.events).toEqual([])
    expect(Option.getOrThrow(h.pending()).installation._tag).toBe("Unattempted")
  })
  it("allows an explicit tray action to authorize installation while preserving a hidden window", async () => {
    const h = harness()
    expect(await h.run(installPreparedUpdate({ continuation: { _tag: "Desktop", showWindow: false }, allowAuthorizationPrompt: true }), {}, {
      requiresAuthorization: true,
      install: (_archive, _release, continuation) => Effect.sync(() => {
        expect(continuation).toEqual({ _tag: "Desktop", showWindow: false })
        expect(Option.getOrThrow(h.pending()).installation._tag).toBe("Attempted")
        h.events.push("install")
      }),
    })).toBe("Started")
    expect(h.events).toEqual(["verify", "attempt", "install"])
  })
  it("never invokes an installer after failed verification or an unsuccessful attempt write", async () => {
    for (const operation of ["verify", "recordAttempt"] as const) {
      const h = harness()
      expect((await h.run(installPreparedUpdate({ continuation: { _tag: "Desktop", showWindow: true }, allowAuthorizationPrompt: true }).pipe(Effect.either), { [operation]: () => new PreparedUpdateFailed({ message: "injected failure" }) }))._tag).toBe("Left")
      expect(h.events).not.toContain("install")
      if (operation === "verify") expect(h.events).toContain("failure")
      else expect(h.events).not.toContain("failure")
    }
  })
  it("persists a known invocation failure against the same release", async () => {
    const h = harness()
    expect((await h.run(installPreparedUpdate({ continuation: { _tag: "Desktop", showWindow: true }, allowAuthorizationPrompt: true }).pipe(Effect.either), {}, { install: () => new ApplicationUpdateFailed({ message: "System authorization was cancelled" }) }))._tag).toBe("Left")
    expect(h.events).toEqual(["verify", "attempt", "failure"])
    expect(Option.getOrThrow(h.pending()).installation).toEqual({ _tag: "Failed", reason: "System authorization was cancelled" })
  })
})
