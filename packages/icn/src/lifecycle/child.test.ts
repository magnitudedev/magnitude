import { BunContext } from "@effect/platform-bun"
import { ProcessGroupController } from "@magnitudedev/utils/process-groups"
import { ProcessGroupControllerLive } from "@magnitudedev/utils/process-groups/native"
import { Deferred, Duration, Effect, Fiber, Layer, Option, Stream } from "effect"
import { describe, expect, it } from "vitest"
import { IcnChildLaunch, IcnChildSpawner, UnixIcnChildSpawner } from "./child"

const layer = UnixIcnChildSpawner.pipe(Layer.provide(Layer.merge(BunContext.layer, Layer.succeed(ProcessGroupController, ProcessGroupControllerLive))))
const launch = new IcnChildLaunch({ executable: "/bin/sh", arguments: ["-c", "sleep 60 & echo $!; printf diagnostic >&2; wait"], environment: {}, gracefulShutdownTimeout: Duration.seconds(1), forceShutdownTimeout: Duration.seconds(1) })
const absent = (pid: number) => Effect.runPromise(ProcessGroupControllerLive.inspect(pid)).then(value => expect(Option.isNone(value)).toBe(true))

describe.skipIf(process.platform === "win32")("ICN Unix child ownership (real processes)", () => {
  it("preserves separate streams and retires the root and descendant with idempotent shutdown", async () => {
    let pid = 0, descendant = 0
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const children = yield* IcnChildSpawner
      const child = yield* children.spawn(launch)
      pid = child.pid
      descendant = Number(Option.getOrThrow(yield* child.stdout.pipe(Stream.decodeText(), Stream.splitLines, Stream.runHead)))
      expect(descendant).toBeGreaterThan(0)
      expect(Option.getOrThrow(yield* child.stderr.pipe(Stream.decodeText(), Stream.runHead))).toBe("diagnostic")
      yield* child.terminate
      yield* child.terminate
      // The platform reports signal termination as a typed error, not a fabricated exit code.
      const exit = yield* Effect.either(child.exitCode)
      expect(exit._tag).toBe("Left")
      if (exit._tag === "Left") expect(exit.left._tag).toBe("SystemError")
    })).pipe(Effect.provide(layer), Effect.timeout("10 seconds")))
    await absent(pid); await absent(descendant)
  })

  it("scope failure retires the already-acquired child", async () => {
    let pid = 0
    const result = await Effect.runPromise(Effect.either(Effect.scoped(Effect.gen(function* () {
      const children = yield* IcnChildSpawner
      const child = yield* children.spawn(launch); pid = child.pid
      return yield* Effect.fail("startup record rejected")
    })).pipe(Effect.provide(layer), Effect.timeout("10 seconds"))))
    expect(result._tag).toBe("Left")
    await absent(pid)
  })

  it("cancellation after acquisition retires the owned group before returning", async () => {
    let pid = 0
    await Effect.runPromise(Effect.gen(function* () {
      const acquired = yield* Deferred.make<number>()
      const fiber = yield* Effect.fork(Effect.scoped(Effect.gen(function* () {
        const children = yield* IcnChildSpawner
        const child = yield* children.spawn(launch)
        yield* Deferred.succeed(acquired, child.pid)
        return yield* Effect.never
      })).pipe(Effect.provide(layer)))
      pid = yield* Deferred.await(acquired)
      yield* Fiber.interrupt(fiber)
    }).pipe(Effect.timeout("10 seconds")))
    await absent(pid)
  })
})
