import {
  AcnOwnerRecordSchema,
  type AcnOwnerRecord,
} from "@magnitudedev/acn-protocol/coordination"
import { type AcnTarget } from "@magnitudedev/acn-protocol"
import { Duration, Schema } from "effect"
import { AcnHealthObservationSchema, type AcnOwnerObservation } from "./acn-owner-observer"
import { type AcnCandidateLaunchState } from "./acn-candidate-launch-supervisor"
import { AcnCandidateFailureSchema } from "./errors"

const ConvergenceShutdownReasonSchema = Schema.Literal(
  "InvalidHealth",
  "RevisionTooOld",
  "HealthUnavailable",
  "StoppingExpired",
  "SurvivingProcessGroup",
)

export const AcnConvergenceDecisionSchema = Schema.Union(
  Schema.TaggedStruct("Wait", {}),
  Schema.TaggedStruct("PrepareLaunch", {}),
  Schema.TaggedStruct("LaunchCandidate", {}),
  Schema.TaggedStruct("ShutdownDaemon", {
    owner: AcnOwnerRecordSchema,
    reason: ConvergenceShutdownReasonSchema,
  }),
  Schema.TaggedStruct("ShutdownDaemonThenFail", {
    owner: AcnOwnerRecordSchema,
    reason: Schema.Literal("StartupExpired"),
  }),
  Schema.TaggedStruct("ConfirmReady", {
    owner: AcnOwnerRecordSchema,
    observed: AcnHealthObservationSchema,
  }),
  Schema.TaggedStruct("FailCandidate", { failure: AcnCandidateFailureSchema }),
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
  reason: typeof ConvergenceShutdownReasonSchema.Type,
): AcnConvergenceDecision => ({ _tag: "ShutdownDaemon", owner, reason })

export const decideAcnConvergence = (snapshot: AcnConvergenceSnapshot): AcnConvergenceDecision => {
  const { candidate, observation, now } = snapshot

  if (candidate._tag === "Failed") return { _tag: "FailCandidate", failure: candidate.failure }

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
