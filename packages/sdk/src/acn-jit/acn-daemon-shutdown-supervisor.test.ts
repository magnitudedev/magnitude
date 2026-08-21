import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientResponse from "@effect/platform/HttpClientResponse"
import { ProcessStartIdentitySchema } from "@magnitudedev/acn-protocol"
import {
  AcnOwnerStore,
  ProcessGroupController,
  ProcessGroupPresent,
  ProcessGroupSignaled,
  ProcessGroupSignalPermissionDenied,
  type AcnOwnerRecord,
} from "@magnitudedev/acn-protocol/coordination"
import { Duration, Effect, Fiber, Option, Ref, TestClock, TestContext } from "effect"
import { describe, expect, it } from "vitest"
import {
  AcnDaemonShutdownFsm,
  AcnDaemonShutdownObserved,
  makeAcnDaemonShutdownSupervisor,
} from "./acn-daemon-shutdown-supervisor"

const expected: AcnOwnerRecord = {
  pid: 41_001,
  processStartIdentity: ProcessStartIdentitySchema.make("darwin:test-session:expected"),
  port: 49_152,
}
const group = { leader: { pid: expected.pid, processStartIdentity: expected.processStartIdentity } }
const makeOwnerStore = (owner: Ref.Ref<Option.Option<AcnOwnerRecord>>): AcnOwnerStore =>
  AcnOwnerStore.of({
    current: Ref.get(owner),
    replaceOwner: () => Effect.dieMessage("shutdown supervisor must not replace owners"),
  })

describe("AcnDaemonShutdownSupervisor", () => {
  it("classifies EPERM as a terminal permission denial instead of leaking a platform exception", async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const owner = yield* Ref.make(Option.some(expected))
      let shutdownRequests = 0
      const http = HttpClient.make((request) => Effect.sync(() => {
        shutdownRequests += 1
        return HttpClientResponse.fromWeb(request, Response.json({}))
      }))
      const processes = ProcessGroupController.of({
        inspect: () => Effect.succeed(Option.some(expected.processStartIdentity)),
        currentProcess: Effect.succeed(expected),
        observeGroup: () => Effect.succeed(new ProcessGroupPresent({ group })),
        signalGroup: () => Effect.fail(new ProcessGroupSignalPermissionDenied({
          group,
          message: "SystemError: kill() failed: EPERM: Operation not permitted",
        })),
      })
      const supervisor = yield* makeAcnDaemonShutdownSupervisor(makeOwnerStore(owner), processes, http)
      const fiber = yield* supervisor.shutdown(expected, "HealthUnavailable").pipe(
        Effect.either,
        Effect.fork,
      )
      while (shutdownRequests === 0) yield* Effect.yieldNow()
      yield* TestClock.adjust(Duration.seconds(6))
      expect(yield* Fiber.join(fiber)).toMatchObject({
        _tag: "Left",
        left: {
          _tag: "AcnDaemonProcessGroupTerminationPermissionDenied",
          message: "SystemError: kill() failed: EPERM: Operation not permitted",
        },
      })
    }).pipe(Effect.provide(TestContext.TestContext)))
  })

  it("does not signal after the complete owner row changes", async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const owner = yield* Ref.make(Option.some(expected))
      let shutdownRequests = 0
      let signals = 0
      const replacement = { ...expected, port: expected.port + 1 }
      const http = HttpClient.make((request) => Ref.set(owner, Option.some(replacement)).pipe(
        Effect.tap(() => Effect.sync(() => { shutdownRequests += 1 })),
        Effect.as(HttpClientResponse.fromWeb(request, Response.json({}))),
      ))
      const processes = ProcessGroupController.of({
        inspect: () => Effect.succeed(Option.some(expected.processStartIdentity)),
        currentProcess: Effect.succeed(expected),
        observeGroup: () => Effect.succeed(new ProcessGroupPresent({ group })),
        signalGroup: () => Effect.sync(() => {
          signals += 1
          return new ProcessGroupSignaled({ group })
        }),
      })
      const supervisor = yield* makeAcnDaemonShutdownSupervisor(makeOwnerStore(owner), processes, http)
      const fiber = yield* supervisor.shutdown(expected, "HealthUnavailable").pipe(Effect.fork)
      while (shutdownRequests === 0) yield* Effect.yieldNow()
      yield* TestClock.adjust(Duration.seconds(6))
      expect(yield* Fiber.join(fiber)).toMatchObject({ _tag: "Superseded", cause: "OwnerChanged" })
      expect(signals).toBe(0)
    }).pipe(Effect.provide(TestContext.TestContext)))
  })

  it("declares the shutdown transition graph through the shared FSM utility", () => {
    const observed = new AcnDaemonShutdownObserved({
      expectedOwner: expected,
      shutdownReason: "HealthUnavailable",
    })
    expect(AcnDaemonShutdownFsm.transition(observed, "GracefulShutdownRequested", {})._tag)
      .toBe("GracefulShutdownRequested")
    expect(AcnDaemonShutdownFsm.canTransition("KillRequested", "TerminationRequested")).toBe(false)
    expect(AcnDaemonShutdownFsm.getTerminalStates()).toEqual([
      "AlreadyAbsent",
      "GracefullyStopped",
      "Terminated",
      "Killed",
      "Superseded",
      "AcnDaemonOwnerObservationFailed",
      "AcnDaemonIdentityObservationFailed",
      "AcnDaemonProcessGroupObservationFailed",
      "AcnDaemonProcessGroupTerminationPermissionDenied",
      "AcnDaemonProcessGroupTerminationFailed",
      "AcnDaemonProcessGroupKillPermissionDenied",
      "AcnDaemonProcessGroupKillFailed",
      "AcnDaemonProcessGroupAbsenceUnproven",
    ])
  })
})
