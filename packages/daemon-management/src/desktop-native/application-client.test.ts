import { NodeContext } from "@effect/platform-node"
import { FileSystem } from "@effect/platform"
import { Effect, Fiber, Ref, Schedule, Schema } from "effect"
import { describe, expect, it } from "vitest"
import { ApplicationSnapshot } from "@magnitudedev/sdk/desktop-host"
import { ApplicationControlUnavailable } from "./application-control"
import { launchApplicationProcess, makeApplicationClient } from "./application-client"

const ready = Schema.decodeUnknownSync(ApplicationSnapshot)({
  version: 1, pid: 42, endpoint: "http://127.0.0.1:11101",
  tray: { _tag: "Registered" },
  service: { _tag: "Ready", health: { service: "magnitude-acn", version: "0.0.14", revision: 1, id: "child", pid: 43, rpcVersion: 1, state: { _tag: "Ready" } } },
})
const starting = Schema.decodeUnknownSync(ApplicationSnapshot)({ ...ready, service: { _tag: "Starting", attempt: 1 } })
const stopped = Schema.decodeUnknownSync(ApplicationSnapshot)({ ...ready, service: { _tag: "Stopping" } })

describe("desktop client startup", () => {
  it("observes an existing application without launching or showing another instance", async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const launches = yield* Ref.make(0)
      const intents = yield* Ref.make<string[]>([])
      const client = makeApplicationClient({ launch: (_, observe) => Ref.update(launches, x => x + 1).pipe(Effect.zipRight(observe)),
        request: intent => Ref.update(intents, values => [...values, intent]).pipe(Effect.as(ready)) })
      expect(yield* client.awaitReady(yield* client.ensure(), 1)).toEqual(ready)
      expect(yield* Ref.get(launches)).toBe(0)
      expect(yield* Ref.get(intents)).toEqual(["EnsureRunning"])
    }))
  })
  it("launches once with background intent when the application is absent", async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const launches = yield* Ref.make<string[]>([])
      const client = makeApplicationClient({ launch: (intent, observe) => Ref.update(launches, xs => [...xs, intent]).pipe(Effect.zipRight(observe)),
        request: () => Ref.get(launches).pipe(Effect.flatMap(xs => xs.length ? Effect.succeed(ready) : new ApplicationControlUnavailable({ message: "absent" }))) })
      expect((yield* client.ensure()).pid).toBe(42)
      expect(yield* Ref.get(launches)).toEqual(["EnsureRunning"])
    }))
  })
  it("does not relaunch when the owner quits during startup", async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const launches = yield* Ref.make(0)
      const client = makeApplicationClient({ launch: (_, observe) => Ref.update(launches, x => x + 1).pipe(Effect.zipRight(observe)), request: () => Effect.succeed(stopped) })
      const result = yield* client.awaitReady(starting, 1).pipe(Effect.either)
      expect(result._tag).toBe("Left")
      if (result._tag === "Left") expect(result.left.message).toContain("will not restart")
      expect(yield* Ref.get(launches)).toBe(0)
    }))
  })
  it("rejects a different owner and an incompatible protocol", async () => {
    const client = makeApplicationClient({ launch: (_, observe) => observe, request: () => Effect.succeed({ ...ready, pid: 99 }) })
    const changed = await Effect.runPromise(client.awaitReady(starting, 1).pipe(Effect.either))
    expect(changed._tag).toBe("Left")
    if (changed._tag === "Left") expect(changed.left.message).toContain("changed")
    const incompatible = await Effect.runPromise(client.awaitReady(ready, 2).pipe(Effect.either))
    expect(incompatible._tag).toBe("Left")
    if (incompatible._tag === "Left") expect(incompatible.left.message).toContain("protocol versions")
  })
  it("rejects background and Open demand during shutdown without launching", async () => {
    for (const _tag of ["Stopping", "Stopped"] as const) {
      const snapshot = Schema.decodeUnknownSync(ApplicationSnapshot)({ ...ready, service: { _tag } })
      const client = makeApplicationClient({ launch: () => Effect.die("Must not launch during Quit"), request: () => Effect.succeed(snapshot) })
      for (const intent of ["EnsureRunning", "ShowWindow"] as const) {
        const result = await Effect.runPromise(client.ensure(intent).pipe(Effect.either))
        expect(result._tag).toBe("Left")
        if (result._tag === "Left") expect(result.left.message).toContain("will not restart")
      }
    }
  })
})


describe.skipIf(process.platform === "win32")("detached launch observation", () => {
  it("reports installation rejection without waiting for application control", async () => {
    const result = await Effect.runPromise(launchApplicationProcess({ executable: "/bin/sh", arguments: ["-c", "exit 75"], environment: process.env, installationGuarded: true }, Effect.never).pipe(Effect.timeout("2 seconds"), Effect.either))
    expect(result._tag).toBe("Left")
    if (result._tag === "Left") {
      expect(result.left._tag).toBe("ApplicationLaunchFailed")
      expect("message" in result.left && result.left.message).toContain("package-manager repair")
    }
  })
  it("allows a successful dispatcher to exit before the application responds", async () => {
    const result = await Effect.runPromise(launchApplicationProcess({ executable: "/bin/sh", arguments: ["-c", "exit 0"], environment: process.env }, Effect.sleep("50 millis").pipe(Effect.as(ready))))
    expect(result).toEqual(ready)
  })
  it("does not kill an already spawned app when the requesting command is canceled", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const directory = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-launch-observation-" })
      const pidPath = `${directory}/pid`
      const fiber = yield* launchApplicationProcess({ executable: "/bin/sh", arguments: ["-c", 'echo $$ > "$1"; exec sleep 5', "launch-test", pidPath], environment: process.env }, Effect.never).pipe(Effect.forkScoped)
      const pid = Number(yield* fs.readFileString(pidPath).pipe(Effect.retry(Schedule.spaced("10 millis")), Effect.timeout("2 seconds")))
      yield* Effect.addFinalizer(() => Effect.sync(() => { try { process.kill(pid, "SIGTERM") } catch {} }))
      yield* Fiber.interrupt(fiber)
      expect(() => process.kill(pid, 0)).not.toThrow()
    })).pipe(Effect.provide(NodeContext.layer)))
  })
})
