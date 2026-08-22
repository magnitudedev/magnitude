/**
 * Process-group coordination contract: observations, outcomes, and the
 * `ProcessGroupController` service. Platform-free; the node implementation
 * lives in `./exact-process` (`@magnitudedev/acn-protocol/coordination/exact-process`).
 */
import { Context, Duration, type Effect, type Option, Schema } from "effect"
import { ExactProcessSchema, ProcessGroupSchema } from "./schemas"
import type {
  ExactProcessIdentityObservationFailed,
  ProcessGroupObservationFailed,
  ProcessGroupStopError,
} from "./errors"
import type { ExactProcess, ProcessGroup } from "./schemas"

/** Shared timing for process-group retirement. */
export const PROCESS_GROUP_EXIT_POLL_INTERVAL = Duration.millis(50)
export const PROCESS_GROUP_TERM_WAIT = Duration.seconds(2)
export const PROCESS_GROUP_KILL_WAIT = Duration.seconds(2)

/** The recorded leader occurrence is alive. */
export class ProcessGroupLeaderLive extends Schema.TaggedClass<ProcessGroupLeaderLive>()(
  "ProcessGroupLeaderLive",
  { group: ProcessGroupSchema },
) {}
/** The leader pid is occupied by a different process occurrence; the recorded group may still have members. */
export class ProcessGroupLeaderReplaced extends Schema.TaggedClass<ProcessGroupLeaderReplaced>()(
  "ProcessGroupLeaderReplaced",
  { group: ProcessGroupSchema, observedLeader: ExactProcessSchema },
) {}
/** The leader occurrence is gone but descendants of the group remain. */
export class ProcessGroupSurvivorsOnly extends Schema.TaggedClass<ProcessGroupSurvivorsOnly>()(
  "ProcessGroupSurvivorsOnly",
  { group: ProcessGroupSchema },
) {}
/** No member of the group remains. */
export class ProcessGroupAbsent extends Schema.TaggedClass<ProcessGroupAbsent>()(
  "ProcessGroupAbsent",
  { group: ProcessGroupSchema },
) {}
export type ProcessGroupObservation =
  | ProcessGroupLeaderLive
  | ProcessGroupLeaderReplaced
  | ProcessGroupSurvivorsOnly
  | ProcessGroupAbsent
export type ProcessGroupObservationError =
  | ExactProcessIdentityObservationFailed
  | ProcessGroupObservationFailed

export class ProcessGroupStopped extends Schema.TaggedClass<ProcessGroupStopped>()(
  "ProcessGroupStopped",
  { group: ProcessGroupSchema },
) {}
export type ProcessGroupStopOutcome = ProcessGroupStopped | ProcessGroupLeaderReplaced

export interface ProcessGroupController {
  /** Detects the exact process currently occupying `pid`, or none if the pid is free. */
  readonly inspect: (
    pid: number,
  ) => Effect.Effect<Option.Option<ExactProcess>, ExactProcessIdentityObservationFailed>
  /** This process's own exact identity. */
  readonly currentProcess: Effect.Effect<ExactProcess, ExactProcessIdentityObservationFailed>
  /** The state of the group led by that exact process occurrence. */
  readonly observe: (
    group: ProcessGroup,
  ) => Effect.Effect<ProcessGroupObservation, ProcessGroupObservationError>
  /** Polls until no member of the group remains or the deadline passes; `true` when it is gone. */
  readonly waitForGroupExit: (
    group: ProcessGroup,
    timeout: Duration.DurationInput,
  ) => Effect.Effect<boolean, ProcessGroupObservationFailed>
  /**
   * Retires the group: TERM, wait, KILL, wait, prove absence. The leader's exact identity is
   * checked before every signal; a replaced leader ends the retirement as
   * `ProcessGroupLeaderReplaced` because the recorded group can no longer be targeted safely.
   */
  readonly stop: (
    group: ProcessGroup,
  ) => Effect.Effect<ProcessGroupStopOutcome, ProcessGroupStopError>
}

export const ProcessGroupController = Context.GenericTag<ProcessGroupController>(
  "@magnitudedev/acn-protocol/coordination/ProcessGroupController",
)
