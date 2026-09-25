import { mkdtemp, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { fileURLToPath } from "node:url"
import { Deferred, Effect, Exit, Fiber, Option, Ref, Schema, Scope, TestClock, TestContext } from "effect"
import { describe, expect, it } from "vitest"
import { ApplicationSnapshot } from "@magnitudedev/sdk/desktop-host"
import { acquireApplicationMaintenance, acquireApplicationOwner } from "./application-owner"
import { serveApplicationControl } from "./application-control"
import { acquireUpdateInstallationLease } from "./update-installation-lease"
import { nativeHostLayer } from "./index"

const addon = fileURLToPath(new URL(`../../dist/native/${process.platform}-${process.arch}/desktop-host.node`, import.meta.url))
const directory = Effect.acquireRelease(Effect.promise(() => mkdtemp(join(tmpdir(), "mag-own-"))), path => Effect.promise(() => rm(path, { recursive: true, force: true })))
const snapshot = Schema.decodeUnknownSync(ApplicationSnapshot)({ version: 1, pid: process.pid,
  endpoint: "http://127.0.0.1:11101", service: { _tag: "Starting", attempt: 0 }, owner: { _tag: "Headless" } })
const noLogin = () => Effect.die("Unexpected login request")
const noUpdate = () => Effect.die("Unexpected update request")

describe.skipIf(process.platform === "win32")("native application owner arbitration", () => {
  it.each(["Desktop", "Headless"] as const)("identifies %s for both startup and maintenance contention without stopping it", owner => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    const root = yield* directory
    const first = yield* acquireApplicationOwner(root, { _tag: "Headless" })
    if (first._tag !== "Owner") return yield* Effect.die("Expected owner")
    const intents = yield* Ref.make<string[]>([])
    yield* serveApplicationControl(first.socketPath, {
      snapshot: Effect.succeed({ ...snapshot, owner: owner === "Desktop"
        ? { _tag: "Desktop", tray: { _tag: "Registered" } } : { _tag: "Headless" } }),
      login: noLogin, update: noUpdate, dispatch: intent => Ref.update(intents, values => [...values, intent]),
    })
    const startup = yield* acquireApplicationOwner(root, { _tag: "Headless" }).pipe(Effect.flip)
    const maintenance = yield* acquireApplicationMaintenance(root).pipe(Effect.flip)
    expect(startup.message).toBe(owner === "Desktop"
      ? "Magnitude Desktop is still running. Fully quit the desktop app before retrying."
      : "Another `magnitude serve` command is already running. Stop it before starting another.")
    expect(maintenance.message).toBe(startup.message)
    expect(yield* Ref.get(intents)).toEqual(["Observe", "Observe"])
  })).pipe(Effect.provide(nativeHostLayer(addon)))))

  it("excludes owners during finite maintenance and releases afterward", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    const root = yield* directory
    yield* Effect.scoped(Effect.gen(function* () {
      yield* acquireApplicationMaintenance(root)
      expect(yield* acquireApplicationOwner(root, { _tag: "Headless" }).pipe(Effect.isFailure)).toBe(true)
      expect(yield* acquireApplicationMaintenance(root).pipe(Effect.isFailure)).toBe(true)
    }))
    expect((yield* acquireApplicationOwner(root, { _tag: "Headless" }))._tag).toBe("Owner")
  })).pipe(Effect.provide(nativeHostLayer(addon)))))

  it("does not admit maintenance over a running owner or active installation", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    const root = yield* directory
    yield* Effect.scoped(Effect.gen(function* () {
      yield* acquireApplicationOwner(root, { _tag: "Headless" })
      expect(yield* acquireApplicationMaintenance(root).pipe(Effect.isFailure)).toBe(true)
    }))
    yield* Effect.scoped(Effect.gen(function* () {
      yield* acquireUpdateInstallationLease(root)
      expect(yield* Effect.scoped(acquireApplicationMaintenance(root)).pipe(Effect.isFailure)).toBe(true)
    }))
    yield* acquireApplicationMaintenance(root)
  })).pipe(Effect.provide(nativeHostLayer(addon)))))

  it("admits exactly one owner across concurrent headless attempts", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const root = yield* directory
      const results = yield* Effect.all(Array.from({ length: 32 }, () =>
        acquireApplicationOwner(root, { _tag: "Headless" }).pipe(Effect.either)), { concurrency: "unbounded" })
      expect(results.filter(result => result._tag === "Right")).toHaveLength(1)
      expect(results.filter(result => result._tag === "Left" && result.left._tag === "ApplicationOwnershipFailed")).toHaveLength(31)
    })).pipe(Effect.provide(nativeHostLayer(addon))))
  })

  it("bounds waiting for an unresponsive owner without releasing its lock", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const root = yield* directory
      expect((yield* acquireApplicationOwner(root, { _tag: "Headless" }))._tag).toBe("Owner")
      const contender = yield* acquireApplicationOwner(root, { _tag: "Desktop", intent: "ShowWindow" }).pipe(Effect.either, Effect.forkScoped)
      yield* TestClock.adjust("61 seconds")
      const result = yield* Fiber.join(contender)
      expect(result._tag === "Left" && result.left._tag).toBe("ApplicationOwnershipFailed")
      const stillOwned = yield* acquireApplicationOwner(root, { _tag: "Headless" }).pipe(Effect.either)
      expect(stillOwned._tag === "Left" && stillOwned.left._tag).toBe("ApplicationOwnershipFailed")
    })).pipe(Effect.provide([nativeHostLayer(addon), TestContext.TestContext])))
  }, 15000)

  it("refuses a second headless owner without a control endpoint or replacing the first", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const root = yield* directory
      expect((yield* acquireApplicationOwner(root, { _tag: "Headless" }))._tag).toBe("Owner")
      const second = yield* acquireApplicationOwner(root, { _tag: "Headless" }).pipe(Effect.either)
      expect(second._tag === "Left" && second.left._tag).toBe("ApplicationOwnershipFailed")
    })).pipe(Effect.provide(nativeHostLayer(addon))))
  })

  it("forwards desktop intent to the existing desktop without yielding", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const root = yield* directory
      const first = yield* acquireApplicationOwner(root, { _tag: "Desktop", intent: "ShowWindow" })
      if (first._tag !== "Owner") return yield* Effect.die("Expected initial owner")
      const intents = yield* Ref.make<string[]>([])
      yield* serveApplicationControl(first.socketPath, { snapshot: Effect.succeed({ ...snapshot, owner: { _tag: "Desktop", tray: { _tag: "Registered" } } }),
        login: noLogin, update: noUpdate, dispatch: intent => Ref.update(intents, values => [...values, intent]) })
      const second = yield* acquireApplicationOwner(root, { _tag: "Desktop", intent: "ShowWindow" })
      expect(second._tag).toBe("Forwarded")
      expect(yield* Ref.get(intents)).toEqual(["Observe", "ShowWindow"])
    })).pipe(Effect.provide(nativeHostLayer(addon))))
  })

  it("waits for actual lock release after headless Yield acknowledgement", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const root = yield* directory
      const firstScope = yield* Scope.make()
      yield* Effect.addFinalizer(() => Scope.close(firstScope, Exit.void))
      const first = yield* acquireApplicationOwner(root, { _tag: "Headless" }).pipe(Effect.provideService(Scope.Scope, firstScope))
      if (first._tag !== "Owner") return yield* Effect.die("Expected initial owner")
      const yielded = yield* Deferred.make<void>()
      yield* serveApplicationControl(first.socketPath, { snapshot: Effect.succeed(snapshot), login: noLogin, update: noUpdate,
        dispatch: intent => intent === "Yield" ? Deferred.succeed(yielded, undefined).pipe(Effect.asVoid) : Effect.void,
      }).pipe(Effect.provideService(Scope.Scope, firstScope))
      const contender = yield* acquireApplicationOwner(root, { _tag: "Desktop", intent: "ShowWindow" }).pipe(Effect.forkScoped)
      yield* Deferred.await(yielded).pipe(Effect.timeout("2 seconds"))
      expect(Option.isNone(yield* Fiber.poll(contender))).toBe(true)
      yield* Scope.close(firstScope, Exit.void)
      expect((yield* Fiber.join(contender).pipe(Effect.timeout("2 seconds")))._tag).toBe("Owner")
      const third = yield* acquireApplicationOwner(root, { _tag: "Headless" }).pipe(Effect.either)
      expect(third._tag === "Left" && third.left._tag).toBe("ApplicationOwnershipFailed")
    })).pipe(Effect.provide(nativeHostLayer(addon))))
  })
})

it("round trips both owner forms and rejects the obsolete snapshot shape", () => {
  for (const owner of [{ _tag: "Headless" }, { _tag: "Desktop", tray: { _tag: "Registered" } }] as const) {
    const value = { ...snapshot, owner }
    expect(Schema.decodeUnknownSync(ApplicationSnapshot)(Schema.encodeSync(ApplicationSnapshot)(value))).toEqual(value)
  }
  const { owner: _, ...old } = snapshot
  expect(Schema.decodeUnknownEither(ApplicationSnapshot)({ ...old, tray: { _tag: "Registered" } })._tag).toBe("Left")
})
