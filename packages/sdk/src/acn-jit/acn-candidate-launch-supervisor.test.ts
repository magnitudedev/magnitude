import { ProcessStartIdentitySchema } from "@magnitudedev/acn-protocol"
import { ProcessGroupController } from "@magnitudedev/acn-protocol/coordination"
import { Effect, Option } from "effect"
import { describe, expect, it } from "vitest"
import {
  AcnCandidateLaunchFsm,
  AcnCandidateNotLaunched,
  makeAcnCandidateLaunchSupervisor,
} from "./acn-candidate-launch-supervisor"
import { ChildProcessSpawner } from "./child-process"
import { AcnCandidateParentChannelReleaseFailed } from "./errors"

describe("AcnCandidateLaunchSupervisor FSM", () => {
  it("makes admission and terminal cleanup dispositions explicit", () => {
    const process = {
      pid: 41_002,
      processStartIdentity: ProcessStartIdentitySchema.make("darwin:test-session:candidate"),
    }
    const notLaunched = new AcnCandidateNotLaunched({})
    const spawned = AcnCandidateLaunchFsm.transition(notLaunched, "Spawned", { process, launchedAt: 1 })
    const admitted = AcnCandidateLaunchFsm.transition(spawned, "Admitted", { admittedAt: 2 })
    expect(AcnCandidateLaunchFsm.transition(admitted, "Ready", {})._tag).toBe("Ready")
    expect(AcnCandidateLaunchFsm.canTransition("Spawned", "AdmissionExpired")).toBe(true)
    expect(AcnCandidateLaunchFsm.canTransition("Admitted", "AdmissionExpired")).toBe(false)
    expect(AcnCandidateLaunchFsm.getTerminalStates()).toEqual([
      "Ready", "LaunchFailed", "ExitedBeforeAdmission", "ExitedAfterAdmission",
      "AdmissionAcknowledgementLost", "AdmissionExpired", "LostAfterAdmission",
    ])
  })

  it("terminalizes a failed admission handoff as Lost", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const process = {
        pid: 41_002,
        processStartIdentity: ProcessStartIdentitySchema.make("darwin:test-session:candidate"),
      }
      const processes = ProcessGroupController.of({
        inspect: () => Effect.succeed(Option.some(process.processStartIdentity)),
        currentProcess: Effect.succeed(process),
        observeGroup: () => Effect.dieMessage("candidate supervision does not observe process groups"),
        signalGroup: () => Effect.dieMessage("candidate supervision does not signal process groups"),
      })
      let cleanups = 0
      const spawner = ChildProcessSpawner.of({
        spawn: () => Effect.succeed({
          pid: process.pid,
          exited: Effect.never,
          confirmExactProcess: () => Effect.void,
          admit: Effect.fail(new AcnCandidateParentChannelReleaseFailed({
            pid: process.pid,
            message: "parent channel failed",
          })),
          stopAndReap: Effect.sync(() => { cleanups += 1 }),
        }),
      })
      const supervisor = yield* makeAcnCandidateLaunchSupervisor(spawner, processes)
      yield* supervisor.launch(["test-acn"])
      const result = yield* supervisor.reconcile(Option.some({ ...process, port: 49_152 })).pipe(
        Effect.either,
      )
      expect(result._tag).toBe("Left")
      expect(yield* supervisor.state).toMatchObject({
        _tag: "AdmissionAcknowledgementLost",
      })
      expect(cleanups).toBe(1)
    })))
  })
})
