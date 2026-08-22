import { ProcessStartIdentitySchema } from "@magnitudedev/acn-protocol"
import {
  ProcessGroupController,
  ProcessGroupLeaderReplaced,
  ProcessGroupSignalPermissionDenied,
  ProcessGroupStopped,
} from "@magnitudedev/acn-protocol/coordination"
import { Cause, Effect, Exit, Fiber, Option } from "effect"
import { describe, expect, it } from "vitest"
import { scopeAcnCandidate } from "./child-process"
import {
  type AcnCandidateBootstrapProcessExitUnproven,
  AcnCandidateBootstrapProcessStopFailed,
  AcnCandidateParentChannelReleaseFailed,
} from "./errors"

const exact = {
  pid: 42,
  processStartIdentity: ProcessStartIdentitySchema.make("darwin:test-session:candidate"),
}

const controller = (onStop: () => void = () => undefined): ProcessGroupController => ({
  inspect: () => Effect.succeed(Option.none()),
  currentProcess: Effect.succeed(exact),
  observe: () => Effect.dieMessage("candidate cleanup does not observe process groups"),
  waitForGroupExit: () => Effect.dieMessage("candidate cleanup does not wait on process groups"),
  stop: (group) => Effect.sync(() => {
    onStop()
    return new ProcessGroupStopped({ group })
  }),
})

const candidate = (options: {
  readonly releaseParentChannel: Effect.Effect<void, AcnCandidateParentChannelReleaseFailed>
  readonly stopBootstrapProcess: Effect.Effect<
    void,
    AcnCandidateBootstrapProcessStopFailed | AcnCandidateBootstrapProcessExitUnproven
  >
}) => scopeAcnCandidate({
  pid: exact.pid,
  exited: Effect.never,
  ...options,
})

const run = <A, E>(effect: Effect.Effect<A, E, ProcessGroupController>) =>
  Effect.runPromise(effect.pipe(Effect.provideService(ProcessGroupController, controller())))

describe("scopeAcnCandidate", () => {
  it("stops the raw child handle when scope closes before exact identity confirmation", async () => {
    let stops = 0
    await run(Effect.scoped(candidate({
      releaseParentChannel: Effect.void,
      stopBootstrapProcess: Effect.sync(() => { stops += 1 }),
    })))
    expect(stops).toBe(1)
  })

  it("disarms scoped cleanup only after successful admission observation", async () => {
    let releases = 0
    let groupStops = 0
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const child = yield* candidate({
        releaseParentChannel: Effect.sync(() => { releases += 1 }),
        stopBootstrapProcess: Effect.void,
      })
      yield* child.confirmExactProcess(exact)
      yield* child.admit
    })).pipe(Effect.provideService(ProcessGroupController, controller(() => { groupStops += 1 }))))
    expect(releases).toBe(1)
    expect(groupStops).toBe(0)
  })

  it("runs exact process-group cleanup when admission acknowledgement fails", async () => {
    let groupStops = 0
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const child = yield* candidate({
        releaseParentChannel: Effect.fail(new AcnCandidateParentChannelReleaseFailed({
          pid: exact.pid,
          message: "bootstrap pipe failed",
        })),
        stopBootstrapProcess: Effect.void,
      })
      yield* child.confirmExactProcess(exact)
      expect((yield* Effect.either(child.admit))._tag).toBe("Left")
    })).pipe(Effect.provideService(ProcessGroupController, controller(() => { groupStops += 1 }))))
    expect(groupStops).toBe(1)
  })

  it("keeps exact cleanup armed when parent-channel release is interrupted", async () => {
    let groupStops = 0
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const child = yield* candidate({
        releaseParentChannel: Effect.never as Effect.Effect<void, AcnCandidateParentChannelReleaseFailed>,
        stopBootstrapProcess: Effect.void,
      })
      yield* child.confirmExactProcess(exact)
      const admitting = yield* child.admit.pipe(Effect.fork)
      yield* Effect.yieldNow()
      yield* Fiber.interrupt(admitting)
    })).pipe(Effect.provideService(ProcessGroupController, controller(() => { groupStops += 1 }))))
    expect(groupStops).toBe(1)
  })

  it("treats a second admission acknowledgement as a defect", async () => {
    let releases = 0
    await run(Effect.scoped(Effect.gen(function* () {
      const child = yield* candidate({
        releaseParentChannel: Effect.sync(() => { releases += 1 }),
        stopBootstrapProcess: Effect.void,
      })
      yield* child.confirmExactProcess(exact)
      yield* child.admit
      const second = yield* Effect.exit(child.admit)
      expect(Exit.isFailure(second) && Cause.isDie(second.cause)).toBe(true)
    })))
    expect(releases).toBe(1)
  })

  it("returns bootstrap cleanup failure without converting it into a defect", async () => {
    const result = await run(Effect.scoped(Effect.gen(function* () {
      const child = yield* candidate({
        releaseParentChannel: Effect.void,
        stopBootstrapProcess: Effect.fail(new AcnCandidateBootstrapProcessStopFailed({
          pid: exact.pid,
          message: "denied",
        })),
      })
      return yield* Effect.either(child.stopAndReap)
    })))
    expect(result).toMatchObject({
      _tag: "Left",
      left: { _tag: "AcnCandidateBootstrapProcessStopFailed", message: "denied" },
    })
  })

  it("preserves process-group signal permission denial", async () => {
    const denied: ProcessGroupController = {
      ...controller(),
      stop: (group) => Effect.fail(new ProcessGroupSignalPermissionDenied({
        group,
        message: "operation not permitted",
      })),
    }
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const child = yield* candidate({
        releaseParentChannel: Effect.void,
        stopBootstrapProcess: Effect.void,
      })
      yield* child.confirmExactProcess(exact)
      return yield* Effect.either(child.stopAndReap)
    })).pipe(Effect.provideService(ProcessGroupController, denied)))
    expect(result).toMatchObject({
      _tag: "Left",
      left: {
        _tag: "ProcessGroupSignalPermissionDenied",
        message: "operation not permitted",
      },
    })
  })

  it("treats a changed group leader as proof the exact child is already gone", async () => {
    const changed: ProcessGroupController = {
      ...controller(),
      stop: (group) => Effect.succeed(new ProcessGroupLeaderReplaced({
        group,
        observedLeader: {
          pid: exact.pid,
          processStartIdentity: ProcessStartIdentitySchema.make("darwin:test-session:reused"),
        },
      })),
    }
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const child = yield* candidate({
        releaseParentChannel: Effect.void,
        stopBootstrapProcess: Effect.void,
      })
      yield* child.confirmExactProcess(exact)
      return yield* Effect.either(child.stopAndReap)
    })).pipe(Effect.provideService(ProcessGroupController, changed)))
    expect(result._tag).toBe("Right")
  })
})
