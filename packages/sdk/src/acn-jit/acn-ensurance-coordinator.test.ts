import { AcnIdentitySchema, AcnRevisionSchema, ProcessStartIdentitySchema } from "@magnitudedev/acn-protocol"
import { ProcessGroupSignalPermissionDenied } from "@magnitudedev/acn-protocol/coordination"
import { Effect, Stream } from "effect"
import { describe, expect, it } from "vitest"
import { AcnCandidateNotLaunched, type AcnCandidateLaunchSupervisor } from "./acn-candidate-launch-supervisor"
import { AcnDaemonShutdownFailed } from "./errors"
import type { AcnDaemonShutdownSupervisor } from "./acn-daemon-shutdown-supervisor"
import { makeAcnEnsuranceCoordinator } from "./acn-ensurance-coordinator"
import type { AcnDaemonLaunchCommandResolver } from "./acn-daemon-launch-command-resolver"
import { AcnRecordedOwnerProcessGroupSurvives, type AcnOwnerObserver } from "./acn-owner-observer"

describe("AcnEnsuranceCoordinator", () => {
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
