import { ProcessStartIdentitySchema } from "@magnitudedev/acn-protocol"
import type { ProcessGroupController } from "@magnitudedev/acn-protocol/coordination"
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
  it("admits one terminal Failed disposition from every live state", () => {
    const process = {
      pid: 41_002,
      processStartIdentity: ProcessStartIdentitySchema.make("darwin:test-session:candidate"),
    }
    const notLaunched = new AcnCandidateNotLaunched({})
    const spawned = AcnCandidateLaunchFsm.transition(notLaunched, "Spawned", { process, launchedAt: 1 })
    const admitted = AcnCandidateLaunchFsm.transition(spawned, "Admitted", { admittedAt: 2 })
    expect(AcnCandidateLaunchFsm.transition(admitted, "Ready", {})._tag).toBe("Ready")
    expect(AcnCandidateLaunchFsm.canTransition("NotLaunched", "Failed")).toBe(true)
    expect(AcnCandidateLaunchFsm.canTransition("Spawned", "Failed")).toBe(true)
    expect(AcnCandidateLaunchFsm.canTransition("Admitted", "Failed")).toBe(true)
    expect(AcnCandidateLaunchFsm.canTransition("Failed", "Spawned")).toBe(false)
    expect(AcnCandidateLaunchFsm.getTerminalStates()).toEqual(["Ready", "Failed"])
  })

  it("terminalizes a failed admission handoff as Failed with the release failure", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const process = {
        pid: 41_002,
        processStartIdentity: ProcessStartIdentitySchema.make("darwin:test-session:candidate"),
      }
      const processes: ProcessGroupController = {
        inspect: () => Effect.succeed(Option.some(process)),
        currentProcess: Effect.succeed(process),
        observe: () => Effect.dieMessage("candidate supervision does not observe process groups"),
        waitForGroupExit: () => Effect.dieMessage("candidate supervision does not wait on process groups"),
        stop: () => Effect.dieMessage("candidate supervision does not stop process groups"),
      }
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
      const result = yield* supervisor.reconcile(Option.some({ ...process, port: 49_152 }))
      expect(result).toMatchObject({
        _tag: "Failed",
        failure: {
          _tag: "AcnCandidateParentChannelReleaseFailed",
          pid: process.pid,
          message: "parent channel failed",
        },
      })
      expect(yield* supervisor.state).toMatchObject({ _tag: "Failed" })
      expect(cleanups).toBe(1)
    })))
  })
})
