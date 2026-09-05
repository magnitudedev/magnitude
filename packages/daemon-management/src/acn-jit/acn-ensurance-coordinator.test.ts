import { AcnIdentitySchema, AcnRevisionSchema, ProcessStartIdentitySchema } from "@magnitudedev/acn-protocol"
import { ProcessGroupSignalPermissionDenied } from "@magnitudedev/acn-protocol/coordination"
import { Effect, Option, Stream } from "effect"
import { describe, expect, it } from "vitest"
import {
  AcnCandidateFailed,
  AcnCandidateNotLaunched,
  type AcnCandidateLaunchSupervisor,
} from "./acn-candidate-launch-supervisor"
import { AcnCandidateOwnershipLost, AcnDaemonShutdownFailed } from "./errors"
import type { AcnDaemonShutdownSupervisor } from "./acn-daemon-shutdown-supervisor"
import { makeAcnEnsuranceCoordinator } from "./acn-ensurance-coordinator"
import type { AcnDaemonLaunchCommandResolver } from "./acn-daemon-launch-command-resolver"
import {
  AcnRecordedOwnerAbsent,
  AcnRecordedOwnerProcessGroupSurvives,
  type AcnOwnerObserver,
} from "./acn-owner-observer"

describe("AcnEnsuranceCoordinator", () => {
  it("does not present a recorded-but-absent owner as live to candidate supervision", async () => {
    const owner = {
      pid: 41_003,
      processStartIdentity: ProcessStartIdentitySchema.make("darwin:test-session:dead-owner"),
      port: 49_152,
    }
    const failure = new AcnCandidateOwnershipLost({ pid: owner.pid })
    const failed = new AcnCandidateFailed({ failure })
    let reconciledOwner: Option.Option<typeof owner> = Option.some(owner)
    const coordinator = await Effect.runPromise(makeAcnEnsuranceCoordinator({
      target: { revision: AcnRevisionSchema.make(1), identity: AcnIdentitySchema.make("test") },
      emit: () => undefined,
      debug: false,
      dataDirectory: "/not-used",
      ownerObserver: {
        observe: Effect.succeed(new AcnRecordedOwnerAbsent({ expectedOwner: Option.some(owner) })),
        confirmReady: () => Effect.dieMessage("an absent owner cannot be ready"),
      },
      shutdownSupervisor: {
        shutdown: () => Effect.dieMessage("an absent owner must not be shut down"),
      },
      candidateSupervisor: {
        state: Effect.succeed(failed),
        changes: Stream.empty,
        launch: () => Effect.dieMessage("a failed candidate must not relaunch"),
        reconcile: (observed) => Effect.sync(() => {
          reconciledOwner = observed
          return failed
        }),
        markReady: () => Effect.dieMessage("a failed candidate cannot be ready"),
      },
      launchCommandResolver: {
        resolve: () => Effect.dieMessage("a failed candidate must not resolve launch material"),
      },
    }))

    const result = await Effect.runPromise(coordinator.run.pipe(Effect.either))
    expect(Option.isNone(reconciledOwner)).toBe(true)
    expect(result).toMatchObject({ _tag: "Left", left: failure })
  })

  it("preserves the concrete shutdown error in its typed failure channel", async () => {
    const owner = {
      pid: 41_004,
      processStartIdentity: ProcessStartIdentitySchema.make("darwin:test-session:owner"),
      port: 49_152,
    }
    const candidateState = new AcnCandidateNotLaunched({})
    const ownerObserver: AcnOwnerObserver = {
      observe: Effect.succeed(new AcnRecordedOwnerProcessGroupSurvives({ owner })),
      confirmReady: () => Effect.dieMessage("process-group survival cannot be ready"),
    }
    const permissionDenied = new AcnDaemonShutdownFailed({
      owner,
      reason: "SurvivingProcessGroup",
      failure: new ProcessGroupSignalPermissionDenied({
        group: { leader: { pid: owner.pid, processStartIdentity: owner.processStartIdentity } },
        message: "Operation not permitted",
      }),
    })
    const shutdownSupervisor: AcnDaemonShutdownSupervisor = {
      shutdown: () => Effect.fail(permissionDenied),
    }
    const candidateSupervisor: AcnCandidateLaunchSupervisor = {
      state: Effect.succeed(candidateState),
      changes: Stream.empty,
      launch: () => Effect.dieMessage("shutdown decision must not launch"),
      reconcile: () => Effect.succeed(candidateState),
      markReady: () => Effect.dieMessage("shutdown decision cannot be ready"),
    }
    const launchCommandResolver: AcnDaemonLaunchCommandResolver = {
      resolve: () => Effect.dieMessage("shutdown decision must not resolve launch material"),
    }
    const coordinator = await Effect.runPromise(makeAcnEnsuranceCoordinator({
      target: { revision: AcnRevisionSchema.make(1), identity: AcnIdentitySchema.make("test") },
      emit: () => undefined,
      debug: false,
      dataDirectory: "/not-used",
      ownerObserver,
      shutdownSupervisor,
      candidateSupervisor,
      launchCommandResolver,
    }))
    const result = await Effect.runPromise(coordinator.run.pipe(Effect.either))
    expect(result).toMatchObject({
      _tag: "Left",
      left: {
        _tag: "AcnDaemonShutdownFailed",
        owner,
        reason: "SurvivingProcessGroup",
        failure: {
          _tag: "ProcessGroupSignalPermissionDenied",
          message: "Operation not permitted",
        },
      },
    })
  })
})
