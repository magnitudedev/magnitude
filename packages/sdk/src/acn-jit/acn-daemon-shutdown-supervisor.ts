import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import { FSM } from "@magnitudedev/utils"
import {
  AcnOwnerRecordSchema,
  sameAcnOwner,
  PROCESS_GROUP_KILL_WAIT,
  PROCESS_GROUP_TERM_WAIT,
  waitForProcessGroupAbsence,
  type AcnOwnerRecord,
  type AcnOwnerStore,
  type AcnOwnerStoreError,
  type ExactProcessIdentityObservationFailed,
  type ProcessGroup,
  type ProcessGroupController,
  type ProcessGroupObservationFailed,
  type ProcessGroupSignalFailed,
  type ProcessGroupSignalPermissionDenied,
} from "@magnitudedev/acn-protocol/coordination"
import { Context, Duration, Effect, Option, Ref, Schema } from "effect"
import {
  AcnDaemonProcessGroupAbsenceUnproven,
  AcnDaemonIdentityObservationFailed,
  AcnDaemonOwnerObservationFailed,
  AcnDaemonShutdownReasonSchema,
  AcnDaemonProcessGroupObservationFailed,
  AcnDaemonProcessGroupKillFailed,
  AcnDaemonProcessGroupKillPermissionDenied,
  AcnDaemonProcessGroupTerminationFailed,
  AcnDaemonProcessGroupTerminationPermissionDenied,
  type AcnDaemonShutdownReason,
} from "./errors"

const GRACEFUL_REQUEST_TIMEOUT = Duration.seconds(2)
const GRACEFUL_STOP_WAIT = Duration.seconds(5)
const ActiveFields = {
  expectedOwner: AcnOwnerRecordSchema,
  shutdownReason: AcnDaemonShutdownReasonSchema,
}

export class AcnDaemonShutdownObserved extends Schema.TaggedClass<AcnDaemonShutdownObserved>()(
  "Observed", ActiveFields,
) {}
export class AcnDaemonGracefulShutdownRequested extends Schema.TaggedClass<AcnDaemonGracefulShutdownRequested>()(
  "GracefulShutdownRequested", ActiveFields,
) {}
export class AcnDaemonTerminationRequested extends Schema.TaggedClass<AcnDaemonTerminationRequested>()(
  "TerminationRequested", ActiveFields,
) {}
export class AcnDaemonKillRequested extends Schema.TaggedClass<AcnDaemonKillRequested>()(
  "KillRequested", ActiveFields,
) {}
export class AcnDaemonAlreadyAbsent extends Schema.TaggedClass<AcnDaemonAlreadyAbsent>()(
  "AlreadyAbsent", ActiveFields,
) {}
export class AcnDaemonGracefullyStopped extends Schema.TaggedClass<AcnDaemonGracefullyStopped>()(
  "GracefullyStopped", ActiveFields,
) {}
export class AcnDaemonTerminated extends Schema.TaggedClass<AcnDaemonTerminated>()(
  "Terminated", ActiveFields,
) {}
export class AcnDaemonKilled extends Schema.TaggedClass<AcnDaemonKilled>()(
  "Killed", ActiveFields,
) {}
export class AcnDaemonSuperseded extends Schema.TaggedClass<AcnDaemonSuperseded>()(
  "Superseded", {
    ...ActiveFields,
    cause: Schema.Literal("OwnerChanged", "IdentityChanged"),
  },
) {}

export const AcnDaemonShutdownStateSchema = Schema.Union(
  AcnDaemonShutdownObserved,
  AcnDaemonGracefulShutdownRequested,
  AcnDaemonTerminationRequested,
  AcnDaemonKillRequested,
  AcnDaemonAlreadyAbsent,
  AcnDaemonGracefullyStopped,
  AcnDaemonTerminated,
  AcnDaemonKilled,
  AcnDaemonSuperseded,
  AcnDaemonOwnerObservationFailed,
  AcnDaemonIdentityObservationFailed,
  AcnDaemonProcessGroupObservationFailed,
  AcnDaemonProcessGroupTerminationPermissionDenied,
  AcnDaemonProcessGroupTerminationFailed,
  AcnDaemonProcessGroupKillPermissionDenied,
  AcnDaemonProcessGroupKillFailed,
  AcnDaemonProcessGroupAbsenceUnproven,
)
export type AcnDaemonShutdownState = typeof AcnDaemonShutdownStateSchema.Type
export type AcnDaemonShutdownFailure =
  | AcnDaemonOwnerObservationFailed
  | AcnDaemonIdentityObservationFailed
  | AcnDaemonProcessGroupObservationFailed
  | AcnDaemonProcessGroupTerminationPermissionDenied
  | AcnDaemonProcessGroupTerminationFailed
  | AcnDaemonProcessGroupKillPermissionDenied
  | AcnDaemonProcessGroupKillFailed
  | AcnDaemonProcessGroupAbsenceUnproven
export type AcnDaemonShutdownSuccess =
  | AcnDaemonAlreadyAbsent
  | AcnDaemonGracefullyStopped
  | AcnDaemonTerminated
  | AcnDaemonKilled
  | AcnDaemonSuperseded
type AcnDaemonShutdownTerminal = AcnDaemonShutdownSuccess | AcnDaemonShutdownFailure

const isAcnDaemonShutdownSuccess = (
  result: AcnDaemonShutdownTerminal,
): result is AcnDaemonShutdownSuccess => {
  switch (result._tag) {
    case "AlreadyAbsent":
    case "GracefullyStopped":
    case "Terminated":
    case "Killed":
    case "Superseded":
      return true
    default:
      return false
  }
}

const terminalFailures = [
  "AcnDaemonOwnerObservationFailed",
  "AcnDaemonIdentityObservationFailed",
  "AcnDaemonProcessGroupObservationFailed",
  "AcnDaemonProcessGroupTerminationPermissionDenied",
  "AcnDaemonProcessGroupTerminationFailed",
  "AcnDaemonProcessGroupKillPermissionDenied",
  "AcnDaemonProcessGroupKillFailed",
  "AcnDaemonProcessGroupAbsenceUnproven",
] as const

export const AcnDaemonShutdownFsm = FSM.defineFSM(
  {
    Observed: AcnDaemonShutdownObserved,
    GracefulShutdownRequested: AcnDaemonGracefulShutdownRequested,
    TerminationRequested: AcnDaemonTerminationRequested,
    KillRequested: AcnDaemonKillRequested,
    AlreadyAbsent: AcnDaemonAlreadyAbsent,
    GracefullyStopped: AcnDaemonGracefullyStopped,
    Terminated: AcnDaemonTerminated,
    Killed: AcnDaemonKilled,
    Superseded: AcnDaemonSuperseded,
    AcnDaemonOwnerObservationFailed,
    AcnDaemonIdentityObservationFailed,
    AcnDaemonProcessGroupObservationFailed,
    AcnDaemonProcessGroupTerminationPermissionDenied,
    AcnDaemonProcessGroupTerminationFailed,
    AcnDaemonProcessGroupKillPermissionDenied,
    AcnDaemonProcessGroupKillFailed,
    AcnDaemonProcessGroupAbsenceUnproven,
  },
  {
    Observed: ["GracefulShutdownRequested", "TerminationRequested", "AlreadyAbsent", "Superseded", ...terminalFailures],
    GracefulShutdownRequested: ["TerminationRequested", "GracefullyStopped", "Superseded", ...terminalFailures],
    TerminationRequested: ["KillRequested", "Terminated", "Superseded", ...terminalFailures],
    KillRequested: ["Killed", "Superseded", ...terminalFailures],
    AlreadyAbsent: [],
    GracefullyStopped: [],
    Terminated: [],
    Killed: [],
    Superseded: [],
    AcnDaemonOwnerObservationFailed: [],
    AcnDaemonIdentityObservationFailed: [],
    AcnDaemonProcessGroupObservationFailed: [],
    AcnDaemonProcessGroupTerminationPermissionDenied: [],
    AcnDaemonProcessGroupTerminationFailed: [],
    AcnDaemonProcessGroupKillPermissionDenied: [],
    AcnDaemonProcessGroupKillFailed: [],
    AcnDaemonProcessGroupAbsenceUnproven: [],
  } as const,
)

export interface AcnDaemonShutdownSupervisor {
  readonly shutdown: (
    expected: AcnOwnerRecord,
    reason: AcnDaemonShutdownReason,
  ) => Effect.Effect<AcnDaemonShutdownSuccess, AcnDaemonShutdownFailure>
}
export const AcnDaemonShutdownSupervisor = Context.GenericTag<AcnDaemonShutdownSupervisor>(
  "@magnitudedev/sdk/AcnDaemonShutdownSupervisor",
)

const groupFrom = (owner: AcnOwnerRecord): ProcessGroup => ({
  leader: { pid: owner.pid, processStartIdentity: owner.processStartIdentity },
})

const storeMessage = (error: AcnOwnerStoreError): string =>
  `${error._tag} at ${error.path}${"message" in error ? `: ${error.message}` : ""}`

export const makeAcnDaemonShutdownSupervisor = (
  owners: AcnOwnerStore,
  processes: ProcessGroupController,
  http: HttpClient.HttpClient,
): Effect.Effect<AcnDaemonShutdownSupervisor> => Effect.gen(function* () {
  const lock = yield* Effect.makeSemaphore(1)

  const shutdown: AcnDaemonShutdownSupervisor["shutdown"] = (expectedOwner, shutdownReason) =>
    Effect.gen(function* () {
      const initial = new AcnDaemonShutdownObserved({ expectedOwner, shutdownReason })
      const state = yield* Ref.make<AcnDaemonShutdownState>(initial)
      const group = groupFrom(expectedOwner)
      const set = <State extends AcnDaemonShutdownState>(next: State): Effect.Effect<State> =>
        Ref.set(state, next).pipe(Effect.as(next))

      const ownerObservationFailed = (
        from: AcnDaemonShutdownState,
        failure: AcnOwnerStoreError,
      ) => set(AcnDaemonShutdownFsm.transition(from, "AcnDaemonOwnerObservationFailed", {
        reason: `Could not revalidate Magnitude daemon owner: ${storeMessage(failure)}`,
        path: failure.path,
        message: storeMessage(failure),
      }))

      const identityObservationFailed = (
        from: AcnDaemonShutdownState,
        failure: ExactProcessIdentityObservationFailed,
      ) => set(AcnDaemonShutdownFsm.transition(from, "AcnDaemonIdentityObservationFailed", {
        reason: `Could not observe Magnitude daemon identity ${expectedOwner.pid}: ${failure.message}`,
        message: failure.message,
      }))

      const processGroupObservationFailed = (
        from: AcnDaemonShutdownState,
        failure: ProcessGroupObservationFailed,
      ) => set(AcnDaemonShutdownFsm.transition(from, "AcnDaemonProcessGroupObservationFailed", {
        reason: `Could not observe Magnitude daemon process group ${expectedOwner.pid}: ${failure.message}`,
        message: failure.message,
      }))

      const processGroupTerminationPermissionDenied = (
        from: AcnDaemonShutdownState,
        failure: ProcessGroupSignalPermissionDenied,
      ) => set(AcnDaemonShutdownFsm.transition(from, "AcnDaemonProcessGroupTerminationPermissionDenied", {
        reason: `Permission denied terminating Magnitude daemon process group ${expectedOwner.pid}: ${failure.message}`,
        message: failure.message,
      }))

      const processGroupTerminationFailed = (
        from: AcnDaemonShutdownState,
        failure: ProcessGroupSignalFailed,
      ) => set(AcnDaemonShutdownFsm.transition(from, "AcnDaemonProcessGroupTerminationFailed", {
        reason: `Failed terminating Magnitude daemon process group ${expectedOwner.pid}: ${failure.message}`,
        message: failure.message,
      }))

      const processGroupKillPermissionDenied = (
        from: AcnDaemonShutdownState,
        failure: ProcessGroupSignalPermissionDenied,
      ) => set(AcnDaemonShutdownFsm.transition(from, "AcnDaemonProcessGroupKillPermissionDenied", {
        reason: `Permission denied killing Magnitude daemon process group ${expectedOwner.pid}: ${failure.message}`,
        message: failure.message,
      }))

      const processGroupKillFailed = (
        from: AcnDaemonShutdownState,
        failure: ProcessGroupSignalFailed,
      ) => set(AcnDaemonShutdownFsm.transition(from, "AcnDaemonProcessGroupKillFailed", {
        reason: `Failed killing Magnitude daemon process group ${expectedOwner.pid}: ${failure.message}`,
        message: failure.message,
      }))

      const revalidate = Effect.gen(function* () {
        const current = yield* owners.current.pipe(Effect.either)
        if (current._tag === "Left") return { _tag: "OwnerObservationFailed", failure: current.left } as const
        if (!Option.exists(current.right, (owner) => sameAcnOwner(owner, expectedOwner))) {
          return { _tag: "Superseded", cause: "OwnerChanged" } as const
        }
        const identity = yield* processes.inspect(expectedOwner.pid).pipe(Effect.either)
        if (identity._tag === "Left") return { _tag: "ProcessObservationFailed", failure: identity.left } as const
        if (Option.isSome(identity.right) && identity.right.value !== expectedOwner.processStartIdentity) {
          return { _tag: "Superseded", cause: "IdentityChanged" } as const
        }
        return { _tag: "Current", rootLive: Option.contains(identity.right, expectedOwner.processStartIdentity) } as const
      })

      const classifyValidation = (
        from: AcnDaemonShutdownState,
        validation: Effect.Effect.Success<typeof revalidate>,
      ): Effect.Effect<AcnDaemonShutdownTerminal | undefined> => {
        switch (validation._tag) {
          case "Superseded":
            return set(AcnDaemonShutdownFsm.transition(from, "Superseded", { cause: validation.cause }))
          case "OwnerObservationFailed":
            return ownerObservationFailed(from, validation.failure)
          case "ProcessObservationFailed":
            return identityObservationFailed(from, validation.failure)
          case "Current":
            return Effect.as(Effect.void, undefined as AcnDaemonShutdownTerminal | undefined)
        }
      }

      const initialGroup = yield* processes.observeGroup(group).pipe(Effect.either)
      if (initialGroup._tag === "Left") {
        return yield* processGroupObservationFailed(initial, initialGroup.left)
      }
      if (initialGroup.right._tag === "ProcessGroupAbsent") {
        return yield* set(AcnDaemonShutdownFsm.transition(initial, "AlreadyAbsent", {}))
      }

      const gracefulAttempt = yield* lock.withPermits(1)(Effect.gen(function* () {
        const validation = yield* revalidate
        if (validation._tag !== "Current" || !validation.rootLive) {
          return { validation, requested: false } as const
        }
        yield* http.execute(HttpClientRequest.post(`http://127.0.0.1:${expectedOwner.port}/shutdown`)).pipe(
          Effect.timeout(GRACEFUL_REQUEST_TIMEOUT),
          Effect.ignore,
        )
        return { validation, requested: true } as const
      }))
      const validationResult = yield* classifyValidation(initial, gracefulAttempt.validation)
      if (validationResult !== undefined) return validationResult

      let current: AcnDaemonShutdownState = initial
      if (gracefulAttempt.requested) {
        current = yield* set(AcnDaemonShutdownFsm.transition(initial, "GracefulShutdownRequested", {}))
        const groupAfterGraceful = yield* waitForProcessGroupAbsence(
          processes,
          group,
          GRACEFUL_STOP_WAIT,
        ).pipe(Effect.either)
        if (groupAfterGraceful._tag === "Left") {
          return yield* processGroupObservationFailed(current, groupAfterGraceful.left)
        }
        if (groupAfterGraceful.right._tag === "ProcessGroupAbsent") {
          return yield* set(AcnDaemonShutdownFsm.transition(current, "GracefullyStopped", {}))
        }
      }

      const termAttempt = yield* lock.withPermits(1)(Effect.gen(function* () {
        const validation = yield* revalidate
        if (validation._tag !== "Current") return { _tag: "Validation", validation } as const
        return { _tag: "Signal", result: yield* processes.signalGroup(group, "term").pipe(Effect.either) } as const
      }))
      if (termAttempt._tag === "Validation") {
        const termValidationResult = yield* classifyValidation(current, termAttempt.validation)
        if (termValidationResult !== undefined) return termValidationResult
      } else if (termAttempt.result._tag === "Left") {
        switch (termAttempt.result.left._tag) {
          case "ExactProcessIdentityObservationFailed":
            return yield* identityObservationFailed(current, termAttempt.result.left)
          case "ProcessGroupSignalPermissionDenied":
            return yield* processGroupTerminationPermissionDenied(current, termAttempt.result.left)
          case "ProcessGroupSignalFailed":
            return yield* processGroupTerminationFailed(current, termAttempt.result.left)
        }
      } else if (termAttempt.result.right._tag === "ProcessGroupLeaderChanged") {
        return yield* set(AcnDaemonShutdownFsm.transition(current, "Superseded", { cause: "IdentityChanged" }))
      } else if (termAttempt.result.right._tag === "ProcessGroupAlreadyAbsent") {
        return current._tag === "GracefulShutdownRequested"
          ? yield* set(AcnDaemonShutdownFsm.transition(current, "GracefullyStopped", {}))
          : yield* set(AcnDaemonShutdownFsm.transition(current, "AlreadyAbsent", {}))
      }
      current = yield* set(AcnDaemonShutdownFsm.transition(current, "TerminationRequested", {}))
      const groupAfterTerm = yield* waitForProcessGroupAbsence(
        processes,
        group,
        PROCESS_GROUP_TERM_WAIT,
      ).pipe(Effect.either)
      if (groupAfterTerm._tag === "Left") {
        return yield* processGroupObservationFailed(current, groupAfterTerm.left)
      }
      if (groupAfterTerm.right._tag === "ProcessGroupAbsent") {
        return yield* set(AcnDaemonShutdownFsm.transition(current, "Terminated", {}))
      }

      const killAttempt = yield* lock.withPermits(1)(Effect.gen(function* () {
        const validation = yield* revalidate
        if (validation._tag !== "Current") return { _tag: "Validation", validation } as const
        return { _tag: "Signal", result: yield* processes.signalGroup(group, "kill").pipe(Effect.either) } as const
      }))
      if (killAttempt._tag === "Validation") {
        const killValidationResult = yield* classifyValidation(current, killAttempt.validation)
        if (killValidationResult !== undefined) return killValidationResult
      } else if (killAttempt.result._tag === "Left") {
        switch (killAttempt.result.left._tag) {
          case "ExactProcessIdentityObservationFailed":
            return yield* identityObservationFailed(current, killAttempt.result.left)
          case "ProcessGroupSignalPermissionDenied":
            return yield* processGroupKillPermissionDenied(current, killAttempt.result.left)
          case "ProcessGroupSignalFailed":
            return yield* processGroupKillFailed(current, killAttempt.result.left)
        }
      } else if (killAttempt.result.right._tag === "ProcessGroupLeaderChanged") {
        return yield* set(AcnDaemonShutdownFsm.transition(current, "Superseded", { cause: "IdentityChanged" }))
      } else if (killAttempt.result.right._tag === "ProcessGroupAlreadyAbsent") {
        return yield* set(AcnDaemonShutdownFsm.transition(current, "Terminated", {}))
      }
      current = yield* set(AcnDaemonShutdownFsm.transition(current, "KillRequested", {}))
      const groupAfterKill = yield* waitForProcessGroupAbsence(
        processes,
        group,
        PROCESS_GROUP_KILL_WAIT,
      ).pipe(Effect.either)
      if (groupAfterKill._tag === "Left") {
        return yield* processGroupObservationFailed(current, groupAfterKill.left)
      }
      return groupAfterKill.right._tag === "ProcessGroupAbsent"
        ? yield* set(AcnDaemonShutdownFsm.transition(current, "Killed", {}))
        : yield* set(AcnDaemonShutdownFsm.transition(current, "AcnDaemonProcessGroupAbsenceUnproven", {
            reason: `Could not prove ACN daemon process group ${expectedOwner.pid} absent after KILL`,
          }))
    }).pipe(Effect.flatMap((result) => isAcnDaemonShutdownSuccess(result)
      ? Effect.succeed(result)
      : Effect.fail(result)))

  return AcnDaemonShutdownSupervisor.of({ shutdown })
})

export type { AcnDaemonShutdownReason } from "./errors"
