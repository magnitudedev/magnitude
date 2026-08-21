import {
  AcnOwnerRecordSchema,
  ExactProcessSchema,
  type AcnOwnerRecord,
} from "@magnitudedev/acn-protocol/coordination"
import { type AcnTarget } from "@magnitudedev/acn-protocol"
import { Duration, Schema } from "effect"
import { AcnHealthObservationSchema, type AcnOwnerObservation } from "./acn-owner-observer"
import {
  AcnCandidateLaunchFailureSchema,
  type AcnCandidateLaunchState,
} from "./acn-candidate-launch-supervisor"
import { type AcnDaemonShutdownReason } from "./acn-daemon-shutdown-supervisor"

export const AcnConvergenceDecisionSchema = Schema.Union(
  Schema.TaggedStruct("Wait", {}),
  Schema.TaggedStruct("PrepareLaunch", {}),
  Schema.TaggedStruct("LaunchCandidate", {}),
  Schema.TaggedStruct("ShutdownDaemon", {
    owner: AcnOwnerRecordSchema,
    reason: Schema.Literal(
      "InvalidHealth", "RevisionTooOld", "HealthUnavailable", "StoppingExpired",
      "StartupExpired", "SurvivingProcessGroup", "AdministrativeStop",
    ),
  }),
  Schema.TaggedStruct("ShutdownDaemonThenFail", {
    owner: AcnOwnerRecordSchema,
    reason: Schema.Literal("StartupExpired"),
  }),
  Schema.TaggedStruct("ConfirmReady", {
    owner: AcnOwnerRecordSchema,
    observed: AcnHealthObservationSchema,
  }),
  Schema.TaggedStruct("FailCandidateLaunch", { failure: AcnCandidateLaunchFailureSchema }),
  Schema.TaggedStruct("FailCandidateExitedBeforeAdmission", {
    process: ExactProcessSchema,
    code: Schema.Number,
    stderr: Schema.String,
  }),
  Schema.TaggedStruct("FailCandidateExitedAfterAdmission", {
    process: ExactProcessSchema,
    code: Schema.Number,
    stderr: Schema.String,
  }),
  Schema.TaggedStruct("FailCandidateAdmissionTimedOut", { process: ExactProcessSchema }),
  Schema.TaggedStruct("FailCandidateAdmissionLost", { process: ExactProcessSchema }),
  Schema.TaggedStruct("FailCandidateOwnershipLostAfterAdmission", { process: ExactProcessSchema }),
)
export type AcnConvergenceDecision = typeof AcnConvergenceDecisionSchema.Type

export interface AcnConvergenceSnapshot {
  readonly target: AcnTarget
  readonly observation: AcnOwnerObservation
  readonly candidate: AcnCandidateLaunchState
  readonly launchPrepared: boolean
  readonly now: number
  readonly ownerObservedAt: number
  readonly healthStateObservedAt: number
}

const HEALTH_GRACE = Duration.seconds(30)
const STARTUP_CEILING = Duration.minutes(5)
const STOPPING_GRACE = Duration.seconds(5)

const shutdown = (
  owner: AcnOwnerRecord,
  reason: AcnDaemonShutdownReason,
): AcnConvergenceDecision => ({ _tag: "ShutdownDaemon", owner, reason })

export const decideAcnConvergence = (snapshot: AcnConvergenceSnapshot): AcnConvergenceDecision => {
  const { candidate, observation, now } = snapshot

  if (candidate._tag === "LaunchFailed") return { _tag: "FailCandidateLaunch", failure: candidate.failure }
  if (candidate._tag === "ExitedBeforeAdmission") {
    return {
      _tag: "FailCandidateExitedBeforeAdmission",
      process: candidate.process,
      code: candidate.code,
      stderr: candidate.stderr,
    }
  }
  if (candidate._tag === "ExitedAfterAdmission") {
    return {
      _tag: "FailCandidateExitedAfterAdmission",
      process: candidate.process,
      code: candidate.code,
      stderr: candidate.stderr,
    }
  }
  if (candidate._tag === "AdmissionExpired") {
    return { _tag: "FailCandidateAdmissionTimedOut", process: candidate.process }
  }
  if (candidate._tag === "AdmissionAcknowledgementLost") {
    return { _tag: "FailCandidateAdmissionLost", process: candidate.process }
  }
  if (candidate._tag === "LostAfterAdmission") {
    return { _tag: "FailCandidateOwnershipLostAfterAdmission", process: candidate.process }
  }

  if (observation._tag === "AcnRecordedOwnerAbsent") {
    return candidate._tag === "NotLaunched" ? { _tag: "LaunchCandidate" } : { _tag: "Wait" }
  }
  if (observation._tag === "AcnRecordedOwnerProcessGroupSurvives") {
    return shutdown(observation.owner, "SurvivingProcessGroup")
  }

  const { owner } = observation
  if (observation._tag === "AcnRecordedOwnerLiveWithoutHealth") {
    return now - snapshot.healthStateObservedAt >= Duration.toMillis(HEALTH_GRACE)
      ? shutdown(owner, "HealthUnavailable")
      : { _tag: "Wait" }
  }

  const observed = observation.health
  const state = observed.health.state
  if (observed.health.pid !== owner.pid ||
    (observed.status === 200) !== (state._tag === "Ready") ||
    (observed.status !== 200 && observed.status !== 503)) {
    return shutdown(owner, "InvalidHealth")
  }
  if (observed.health.revision < snapshot.target.revision) {
    return snapshot.launchPrepared ? shutdown(owner, "RevisionTooOld") : { _tag: "PrepareLaunch" }
  }
  if (state._tag === "Ready") return { _tag: "ConfirmReady", owner, observed }
  if (state._tag === "Starting") {
    return now - snapshot.ownerObservedAt >= Duration.toMillis(STARTUP_CEILING)
      ? {
          _tag: "ShutdownDaemonThenFail",
          owner,
          reason: "StartupExpired",
        }
      : { _tag: "Wait" }
  }
  if (state._tag === "Stopping" &&
    now - snapshot.healthStateObservedAt >= Duration.toMillis(STOPPING_GRACE)) {
    return shutdown(owner, "StoppingExpired")
  }
  return { _tag: "Wait" }
}

export const AcnConvergenceDecider = { decide: decideAcnConvergence } as const
