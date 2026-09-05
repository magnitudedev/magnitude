import { ProcessStartIdentitySchema } from "@magnitudedev/acn-protocol"
import type { ProcessGroupController } from "@magnitudedev/acn-protocol/coordination"
import { Deferred, Effect, Option } from "effect"
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
          retireAdmittedGroup: Effect.void,
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

  it("reaps an admitted process group when its root exits before readiness", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const process = {
        pid: 41_003,
        processStartIdentity: ProcessStartIdentitySchema.make("darwin:test-session:admitted-exit"),
      }
      const exited = yield* Deferred.make<{ readonly code: number; readonly stderr: string }>()
      const processes: ProcessGroupController = {
        inspect: () => Effect.succeed(Option.some(process)),
        currentProcess: Effect.succeed(process),
        observe: () => Effect.dieMessage("candidate supervision does not observe process groups"),
        waitForGroupExit: () => Effect.dieMessage("candidate supervision does not wait on process groups"),
        stop: () => Effect.dieMessage("candidate process-group cleanup belongs to the child handle"),
      }
      let reaps = 0
      const spawner = ChildProcessSpawner.of({
        spawn: () => Effect.succeed({
          pid: process.pid,
          exited: Deferred.await(exited),
          confirmExactProcess: () => Effect.void,
          admit: Effect.void,
          stopAndReap: Effect.void,
          retireAdmittedGroup: Effect.sync(() => { reaps += 1 }),
        }),
      })
      const supervisor = yield* makeAcnCandidateLaunchSupervisor(spawner, processes)
      yield* supervisor.launch(["test-acn"])
      const owner = Option.some({ ...process, port: 49_153 })
      expect((yield* supervisor.reconcile(owner))._tag).toBe("Admitted")
      yield* Deferred.succeed(exited, { code: 1, stderr: "public port unavailable" })
      const result = yield* supervisor.reconcile(owner)
      expect(result).toMatchObject({
        _tag: "Failed",
        failure: {
          _tag: "AcnCandidateExitedAfterAdmission",
          code: 1,
          stderr: "public port unavailable",
        },
      })
      expect(reaps).toBe(1)
    })))
  })

  it("retires an admitted process group when ownership is lost before readiness", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const process = {
        pid: 41_004,
        processStartIdentity: ProcessStartIdentitySchema.make("darwin:test-session:ownership-lost"),
      }
      const processes: ProcessGroupController = {
        inspect: () => Effect.succeed(Option.some(process)),
        currentProcess: Effect.succeed(process),
        observe: () => Effect.dieMessage("candidate supervision does not observe process groups"),
        waitForGroupExit: () => Effect.dieMessage("candidate supervision does not wait on process groups"),
        stop: () => Effect.dieMessage("candidate process-group cleanup belongs to the child handle"),
      }
      let retirements = 0
      const spawner = ChildProcessSpawner.of({
        spawn: () => Effect.succeed({
          pid: process.pid,
          exited: Effect.never,
          confirmExactProcess: () => Effect.void,
          admit: Effect.void,
          stopAndReap: Effect.void,
          retireAdmittedGroup: Effect.sync(() => { retirements += 1 }),
        }),
      })
      const supervisor = yield* makeAcnCandidateLaunchSupervisor(spawner, processes)
      yield* supervisor.launch(["test-acn"])
      expect((yield* supervisor.reconcile(Option.some({ ...process, port: 49_154 })))._tag)
        .toBe("Admitted")
      const result = yield* supervisor.reconcile(Option.none())
      expect(result).toMatchObject({
        _tag: "Failed",
        failure: { _tag: "AcnCandidateOwnershipLost", pid: process.pid },
      })
      expect(retirements).toBe(1)
    })))
  })
})
