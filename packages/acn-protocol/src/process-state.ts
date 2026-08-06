import * as FileSystem from "@effect/platform/FileSystem"
import * as Command from "@effect/platform/Command"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import type { PlatformError } from "@effect/platform/Error"
import { randomUUID } from "node:crypto"
import * as NodePath from "node:path"
import { Data, Effect, Option, Schema } from "effect"
import { compareAcnIdentities } from "./acn-identity"
import {
  AcnIdentitySchema,
  AcnInstanceIdSchema,
  ProcessStartIdentitySchema,
} from "./acn-identity"

const PositiveSafeInteger = Schema.Number.pipe(
  Schema.int(),
  Schema.positive(),
  Schema.lessThanOrEqualTo(Number.MAX_SAFE_INTEGER),
)
const NonEmptyString = Schema.String.pipe(Schema.minLength(1))

export const AcnProcessRevisionSchema = PositiveSafeInteger.pipe(
  Schema.brand("AcnProcessRevision"),
)
export type AcnProcessRevision = typeof AcnProcessRevisionSchema.Type

export const ExactProcessSchema = Schema.Struct({
  pid: PositiveSafeInteger,
  processStartIdentity: ProcessStartIdentitySchema,
})
export type ExactProcess = typeof ExactProcessSchema.Type

export const ExactAcnCandidateSchema = Schema.Struct({
  identity: AcnIdentitySchema,
  pid: PositiveSafeInteger,
  processStartIdentity: ProcessStartIdentitySchema,
})
export type ExactAcnCandidate = typeof ExactAcnCandidateSchema.Type

export const AssignedAcnSchema = Schema.Struct({
  id: AcnInstanceIdSchema,
  identity: AcnIdentitySchema,
  url: NonEmptyString,
  pid: PositiveSafeInteger,
  processStartIdentity: ProcessStartIdentitySchema,
})
export type AssignedAcn = typeof AssignedAcnSchema.Type

export const AcnChangeResultSchema = Schema.Union(
  Schema.TaggedStruct("Admitted", { changeRevision: AcnProcessRevisionSchema }),
  Schema.TaggedStruct("Terminated", { changeRevision: AcnProcessRevisionSchema }),
  Schema.TaggedStruct("Failed", {
    changeRevision: AcnProcessRevisionSchema,
    reason: NonEmptyString,
  }),
)
export type AcnChangeResult = typeof AcnChangeResultSchema.Type

const AcnChangePurposeSchema = Schema.Union(
  Schema.TaggedStruct("Ensure", { target: AcnIdentitySchema }),
  Schema.TaggedStruct("Terminate", {}),
)
export type AcnChangePurpose = typeof AcnChangePurposeSchema.Type

const OptionalAssignedAcn = Schema.optionalWith(AssignedAcnSchema, {
  as: "Option",
  exact: true,
})

const ManagerPhaseSchema = Schema.Union(
  Schema.TaggedStruct("Preparing", { current: OptionalAssignedAcn }),
  Schema.TaggedStruct("RetiringAssigned", { current: AssignedAcnSchema }),
  Schema.TaggedStruct("RetiringCandidate", { candidate: ExactAcnCandidateSchema }),
  Schema.TaggedStruct("BlockedCandidateCleanup", {
    candidate: ExactAcnCandidateSchema,
    reason: NonEmptyString,
  }),
  Schema.TaggedStruct("Spawning", {}),
)
export type ManagerPhase = typeof ManagerPhaseSchema.Type

const AcnChangeOwnerSchema = Schema.Union(
  Schema.TaggedStruct("Manager", {
    process: ExactProcessSchema,
    phase: ManagerPhaseSchema,
  }),
  Schema.TaggedStruct("Candidate", { candidate: ExactAcnCandidateSchema }),
)
export type AcnChangeOwner = typeof AcnChangeOwnerSchema.Type

const OptionalResult = Schema.optionalWith(AcnChangeResultSchema, {
  as: "Option",
  exact: true,
})

export const AcnProcessModeSchema = Schema.Union(
  Schema.TaggedStruct("Unassigned", { result: OptionalResult }),
  Schema.TaggedStruct("Assigned", {
    current: AssignedAcnSchema,
    result: OptionalResult,
  }),
  Schema.TaggedStruct("Changing", {
    changeRevision: AcnProcessRevisionSchema,
    purpose: AcnChangePurposeSchema,
    owner: AcnChangeOwnerSchema,
  }),
)
export type AcnProcessMode = typeof AcnProcessModeSchema.Type

export const AcnProcessStateSchema = Schema.Struct({
  revision: AcnProcessRevisionSchema,
  identityFloor: AcnIdentitySchema,
  mode: AcnProcessModeSchema,
})
export type AcnProcessState = typeof AcnProcessStateSchema.Type

export const AcnProcessCommandSchema = Schema.Union(
  Schema.TaggedStruct("BeginEnsure", {
    target: AcnIdentitySchema,
    manager: ExactProcessSchema,
  }),
  Schema.TaggedStruct("BeginReplacement", {
    target: AcnIdentitySchema,
    manager: ExactProcessSchema,
    current: AssignedAcnSchema,
  }),
  Schema.TaggedStruct("BeginTerminate", {
    manager: ExactProcessSchema,
    current: AssignedAcnSchema,
  }),
  Schema.TaggedStruct("BeginTerminateCurrent", {
    manager: ExactProcessSchema,
  }),
  Schema.TaggedStruct("UpgradeEnsure", {
    target: AcnIdentitySchema,
    manager: ExactProcessSchema,
  }),
  Schema.TaggedStruct("TakeOver", { manager: ExactProcessSchema }),
  Schema.TaggedStruct("PreparationSucceeded", { manager: ExactProcessSchema }),
  Schema.TaggedStruct("PreparationFailed", {
    manager: ExactProcessSchema,
    reason: NonEmptyString,
  }),
  Schema.TaggedStruct("PredecessorExited", { manager: ExactProcessSchema }),
  Schema.TaggedStruct("CandidateSpawned", {
    manager: ExactProcessSchema,
    candidate: ExactAcnCandidateSchema,
  }),
  Schema.TaggedStruct("CandidateAdmitted", {
    candidate: ExactAcnCandidateSchema,
    id: AcnInstanceIdSchema,
    url: NonEmptyString,
  }),
  Schema.TaggedStruct("CandidateExited", { manager: ExactProcessSchema }),
  Schema.TaggedStruct("CandidateFailed", {
    candidate: ExactAcnCandidateSchema,
    reason: NonEmptyString,
  }),
  Schema.TaggedStruct("CandidateCleanupBlocked", {
    manager: ExactProcessSchema,
    reason: NonEmptyString,
  }),
  Schema.TaggedStruct("RetryCandidateCleanup", { manager: ExactProcessSchema }),
  Schema.TaggedStruct("FailSpawning", {
    manager: ExactProcessSchema,
    reason: NonEmptyString,
  }),
  Schema.TaggedStruct("RetirementBlocked", {
    manager: ExactProcessSchema,
    reason: NonEmptyString,
  }),
)
export type AcnProcessCommand = typeof AcnProcessCommandSchema.Type

export class AcnProcessStateInvalid extends Data.TaggedError("AcnProcessStateInvalid")<{
  readonly path: string
  readonly reason: string
}> {}

export class AcnProcessStateUnavailable extends Data.TaggedError("AcnProcessStateUnavailable")<{
  readonly path: string
  readonly reason: string
}> {}

export class AcnProcessStateConflict extends Data.TaggedError("AcnProcessStateConflict")<{
  readonly expected: Option.Option<AcnProcessRevision>
  readonly actual: Option.Option<AcnProcessRevision>
}> {}

export class AcnProcessTransitionRejected extends Data.TaggedError("AcnProcessTransitionRejected")<{
  readonly command: AcnProcessCommand["_tag"]
  readonly reason: string
}> {}

export type AcnProcessStateError =
  | AcnProcessStateInvalid
  | AcnProcessStateUnavailable
  | AcnProcessStateConflict
  | AcnProcessTransitionRejected

export class ProcessInspectionFailed extends Data.TaggedError("ProcessInspectionFailed")<{
  readonly pid: number
  readonly reason: string
}> {}

const inspectionCommand = (pid: number): Command.Command =>
  process.platform === "win32"
    ? Command.make(
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        `(Get-Process -Id ${pid} -ErrorAction Stop).StartTime.ToUniversalTime().Ticks`,
      )
    : Command.make("/bin/ps", "-o", "lstart=", "-p", String(pid))

const pidExists = (pid: number): boolean => {
  try {
    process.kill(pid, 0)
    return true
  } catch (error) {
    return !(error instanceof Error && "code" in error && error.code === "ESRCH")
  }
}

export const readProcessStartIdentity = (
  pid: number,
): Effect.Effect<
  Option.Option<typeof ProcessStartIdentitySchema.Type>,
  ProcessInspectionFailed,
  CommandExecutor.CommandExecutor
> => Effect.gen(function* () {
  if (!pidExists(pid)) return Option.none()
  const inspected = yield* Command.string(inspectionCommand(pid)).pipe(
    Effect.timeout("1 second"),
    Effect.map((output) => output.trim()),
    Effect.either,
  )
  if (inspected._tag === "Right" && inspected.right.length > 0) {
    return Option.some(ProcessStartIdentitySchema.make(inspected.right))
  }
  if (!pidExists(pid)) return Option.none()
  return yield* new ProcessInspectionFailed({
    pid,
    reason: inspected._tag === "Left" ? String(inspected.left) : "process inspection returned an empty identity",
  })
})

export const currentProcessStartIdentity = readProcessStartIdentity(process.pid).pipe(
  Effect.flatMap(Option.match({
    onNone: () => Effect.fail(new ProcessInspectionFailed({
      pid: process.pid,
      reason: "current process disappeared during identity inspection",
    })),
    onSome: Effect.succeed,
  })),
)

const sameProcess = (left: ExactProcess, right: ExactProcess): boolean =>
  left.pid === right.pid && left.processStartIdentity === right.processStartIdentity

const sameCandidate = (left: ExactAcnCandidate, right: ExactAcnCandidate): boolean =>
  left.identity === right.identity && sameProcess(left, right)

const sameAssigned = (left: AssignedAcn, right: AssignedAcn): boolean =>
  left.id === right.id &&
  left.identity === right.identity &&
  left.url === right.url &&
  sameProcess(left, right)

function requireTransition(
  condition: boolean,
  command: AcnProcessCommand,
  reason: string,
): asserts condition {
  if (!condition) {
    throw new AcnProcessTransitionRejected({ command: command._tag, reason })
  }
}

const nextRevision = (current: Option.Option<AcnProcessState>): AcnProcessRevision =>
  AcnProcessRevisionSchema.make(Option.match(current, {
    onNone: () => 1,
    onSome: (state) => state.revision + 1,
  }))

const managerPhaseForTakeOver = (owner: AcnChangeOwner): ManagerPhase => {
  if (owner._tag === "Candidate") {
    return { _tag: "RetiringCandidate", candidate: owner.candidate }
  }
  switch (owner.phase._tag) {
    case "Preparing":
      return owner.phase
    case "RetiringAssigned":
      return { _tag: "Preparing", current: Option.some(owner.phase.current) }
    case "Spawning":
      return { _tag: "Preparing", current: Option.none() }
    case "RetiringCandidate":
    case "BlockedCandidateCleanup":
      return owner.phase
  }
}

/** Pure legal-transition boundary for the process-state protocol. */
export const reduceAcnProcessState = (
  current: Option.Option<AcnProcessState>,
  command: AcnProcessCommand,
): AcnProcessState => {
  const revision = nextRevision(current)
  if (Option.isNone(current)) {
    requireTransition(command._tag === "BeginEnsure", command, "only BeginEnsure may create process state")
    return {
      revision,
      identityFloor: command.target,
      mode: {
        _tag: "Changing",
        changeRevision: revision,
        purpose: { _tag: "Ensure", target: command.target },
        owner: {
          _tag: "Manager",
          process: command.manager,
          phase: { _tag: "Preparing", current: Option.none() },
        },
      },
    }
  }

  const state = current.value
  switch (command._tag) {
    case "BeginEnsure": {
      requireTransition(state.mode._tag === "Unassigned", command, "ACN is not unassigned")
      const floor = compareAcnIdentities(command.target, state.identityFloor) >= 0
        ? command.target
        : state.identityFloor
      return {
        revision,
        identityFloor: floor,
        mode: {
          _tag: "Changing",
          changeRevision: revision,
          purpose: { _tag: "Ensure", target: floor },
          owner: {
            _tag: "Manager",
            process: command.manager,
            phase: { _tag: "Preparing", current: Option.none() },
          },
        },
      }
    }
    case "BeginReplacement": {
      requireTransition(
        state.mode._tag === "Assigned" && sameAssigned(state.mode.current, command.current),
        command,
        "exact assigned ACN does not match",
      )
      const floor = compareAcnIdentities(command.target, state.identityFloor) >= 0
        ? command.target
        : state.identityFloor
      return {
        revision,
        identityFloor: floor,
        mode: {
          _tag: "Changing",
          changeRevision: revision,
          purpose: { _tag: "Ensure", target: floor },
          owner: {
            _tag: "Manager",
            process: command.manager,
            phase: { _tag: "Preparing", current: Option.some(state.mode.current) },
          },
        },
      }
    }
    case "BeginTerminate": {
      requireTransition(
        state.mode._tag === "Assigned" && sameAssigned(state.mode.current, command.current),
        command,
        "exact assigned ACN does not match",
      )
      return {
        revision,
        identityFloor: state.identityFloor,
        mode: {
          _tag: "Changing",
          changeRevision: revision,
          purpose: { _tag: "Terminate" },
          owner: {
            _tag: "Manager",
            process: command.manager,
            phase: { _tag: "RetiringAssigned", current: state.mode.current },
          },
        },
      }
    }
    case "BeginTerminateCurrent": {
      requireTransition(
        state.mode._tag === "Assigned" ||
          (state.mode._tag === "Changing" && state.mode.purpose._tag === "Ensure"),
        command,
        "there is no current ensure or assignment to terminate",
      )
      if (state.mode._tag === "Changing") {
        const phase = state.mode.owner._tag === "Candidate"
          ? { _tag: "RetiringCandidate" as const, candidate: state.mode.owner.candidate }
          : state.mode.owner.phase._tag === "BlockedCandidateCleanup"
            ? {
                _tag: "RetiringCandidate" as const,
                candidate: state.mode.owner.phase.candidate,
              }
            : state.mode.owner.phase
        if (phase._tag === "Spawning" || (phase._tag === "Preparing" && Option.isNone(phase.current))) {
          return {
            revision,
            identityFloor: state.identityFloor,
            mode: {
              _tag: "Unassigned",
              result: Option.some({
                _tag: "Terminated",
                changeRevision: state.mode.changeRevision,
              }),
            },
          }
        }
        const terminationPhase = phase._tag === "Preparing"
          ? { _tag: "RetiringAssigned" as const, current: Option.getOrThrow(phase.current) }
          : phase
        return {
          ...state,
          revision,
          mode: {
            _tag: "Changing",
            changeRevision: state.mode.changeRevision,
            purpose: { _tag: "Terminate" },
            owner: { _tag: "Manager", process: command.manager, phase: terminationPhase },
          },
        }
      }
      return {
        revision,
        identityFloor: state.identityFloor,
        mode: {
          _tag: "Changing",
          changeRevision: revision,
          purpose: { _tag: "Terminate" },
          owner: {
            _tag: "Manager",
            process: command.manager,
            phase: { _tag: "RetiringAssigned", current: state.mode.current },
          },
        },
      }
    }
    case "UpgradeEnsure": {
      requireTransition(
        state.mode._tag === "Changing" && state.mode.purpose._tag === "Ensure",
        command,
        "no active ensure to upgrade",
      )
      requireTransition(
        state.mode.owner._tag === "Manager" && state.mode.owner.phase._tag === "Preparing",
        command,
        "only a preparing ensure may be upgraded",
      )
      const activeTarget = state.mode.purpose.target
      requireTransition(
        compareAcnIdentities(command.target, activeTarget) > 0,
        command,
        "upgrade target must exceed the active target",
      )
      return {
        revision,
        identityFloor: command.target,
        mode: {
          _tag: "Changing",
          changeRevision: state.mode.changeRevision,
          purpose: { _tag: "Ensure", target: command.target },
          owner: {
            _tag: "Manager",
            process: command.manager,
            phase: state.mode.owner.phase,
          },
        },
      }
    }
    case "TakeOver": {
      requireTransition(state.mode._tag === "Changing", command, "no active change to take over")
      return {
        ...state,
        revision,
        mode: {
          ...state.mode,
          owner: {
            _tag: "Manager",
            process: command.manager,
            phase: managerPhaseForTakeOver(state.mode.owner),
          },
        },
      }
    }
    case "PreparationSucceeded": {
      requireTransition(
        state.mode._tag === "Changing" &&
        state.mode.purpose._tag === "Ensure" &&
        state.mode.owner._tag === "Manager" &&
        sameProcess(state.mode.owner.process, command.manager) &&
        state.mode.owner.phase._tag === "Preparing",
        command,
        "manager does not own preparation",
      )
      const current = state.mode.owner.phase.current
      return {
        ...state,
        revision,
        mode: {
          ...state.mode,
          owner: {
            _tag: "Manager",
            process: command.manager,
            phase: Option.match(current, {
              onNone: () => ({ _tag: "Spawning" as const }),
              onSome: (assigned) => ({ _tag: "RetiringAssigned" as const, current: assigned }),
            }),
          },
        },
      }
    }
    case "PreparationFailed": {
      requireTransition(
        state.mode._tag === "Changing" &&
        state.mode.purpose._tag === "Ensure" &&
        state.mode.owner._tag === "Manager" &&
        sameProcess(state.mode.owner.process, command.manager) &&
        state.mode.owner.phase._tag === "Preparing",
        command,
        "manager does not own preparation",
      )
      const result = Option.some({
        _tag: "Failed" as const,
        changeRevision: state.mode.changeRevision,
        reason: command.reason,
      })
      return {
        revision,
        identityFloor: state.identityFloor,
        mode: Option.match(state.mode.owner.phase.current, {
          onNone: () => ({ _tag: "Unassigned" as const, result }),
          onSome: (current) => ({ _tag: "Assigned" as const, current, result }),
        }),
      }
    }
    case "PredecessorExited": {
      requireTransition(
        state.mode._tag === "Changing" &&
        state.mode.owner._tag === "Manager" &&
        sameProcess(state.mode.owner.process, command.manager) &&
        state.mode.owner.phase._tag === "RetiringAssigned",
        command,
        "manager does not own predecessor retirement",
      )
      if (state.mode.purpose._tag === "Terminate") {
        return {
          revision,
          identityFloor: state.identityFloor,
          mode: {
            _tag: "Unassigned",
            result: Option.some({ _tag: "Terminated", changeRevision: state.mode.changeRevision }),
          },
        }
      }
      return {
        ...state,
        revision,
        mode: {
          ...state.mode,
          owner: { _tag: "Manager", process: command.manager, phase: { _tag: "Spawning" } },
        },
      }
    }
    case "CandidateSpawned": {
      requireTransition(
        state.mode._tag === "Changing" &&
        state.mode.purpose._tag === "Ensure" &&
        state.mode.owner._tag === "Manager" &&
        sameProcess(state.mode.owner.process, command.manager) &&
        state.mode.owner.phase._tag === "Spawning" &&
        command.candidate.identity === state.mode.purpose.target,
        command,
        "manager does not own spawning for this target",
      )
      return {
        ...state,
        revision,
        mode: { ...state.mode, owner: { _tag: "Candidate", candidate: command.candidate } },
      }
    }
    case "CandidateAdmitted": {
      requireTransition(
        state.mode._tag === "Changing" &&
        state.mode.purpose._tag === "Ensure" &&
        state.mode.owner._tag === "Candidate" &&
        sameCandidate(state.mode.owner.candidate, command.candidate) &&
        command.candidate.identity === state.mode.purpose.target,
        command,
        "exact candidate does not own admission",
      )
      return {
        revision,
        identityFloor: state.identityFloor,
        mode: {
          _tag: "Assigned",
          current: {
            id: command.id,
            identity: command.candidate.identity,
            url: command.url,
            pid: command.candidate.pid,
            processStartIdentity: command.candidate.processStartIdentity,
          },
          result: Option.some({ _tag: "Admitted", changeRevision: state.mode.changeRevision }),
        },
      }
    }
    case "CandidateExited": {
      requireTransition(
        state.mode._tag === "Changing" &&
        state.mode.owner._tag === "Manager" &&
        sameProcess(state.mode.owner.process, command.manager) &&
        state.mode.owner.phase._tag === "RetiringCandidate",
        command,
        "manager does not own candidate retirement",
      )
      if (state.mode.purpose._tag === "Terminate") {
        return {
          revision,
          identityFloor: state.identityFloor,
          mode: {
            _tag: "Unassigned",
            result: Option.some({
              _tag: "Terminated",
              changeRevision: state.mode.changeRevision,
            }),
          },
        }
      }
      return {
        ...state,
        revision,
        mode: {
          ...state.mode,
          owner: {
            _tag: "Manager",
            process: command.manager,
            phase: { _tag: "Preparing", current: Option.none() },
          },
        },
      }
    }
    case "CandidateFailed": {
      requireTransition(
        state.mode._tag === "Changing" &&
        state.mode.owner._tag === "Candidate" &&
        sameCandidate(state.mode.owner.candidate, command.candidate),
        command,
        "exact candidate does not own the active change",
      )
      return {
        revision,
        identityFloor: state.identityFloor,
        mode: {
          _tag: "Unassigned",
          result: Option.some({
            _tag: "Failed",
            changeRevision: state.mode.changeRevision,
            reason: command.reason,
          }),
        },
      }
    }
    case "CandidateCleanupBlocked": {
      requireTransition(
        state.mode._tag === "Changing" &&
        state.mode.owner._tag === "Manager" &&
        sameProcess(state.mode.owner.process, command.manager) &&
        state.mode.owner.phase._tag === "RetiringCandidate",
        command,
        "manager does not own candidate retirement",
      )
      return {
        ...state,
        revision,
        mode: {
          ...state.mode,
          owner: {
            _tag: "Manager",
            process: command.manager,
            phase: {
              _tag: "BlockedCandidateCleanup",
              candidate: state.mode.owner.phase.candidate,
              reason: command.reason,
            },
          },
        },
      }
    }
    case "RetryCandidateCleanup": {
      requireTransition(
        state.mode._tag === "Changing" &&
        state.mode.owner._tag === "Manager" &&
        state.mode.owner.phase._tag === "BlockedCandidateCleanup",
        command,
        "candidate cleanup is not blocked",
      )
      return {
        ...state,
        revision,
        mode: {
          ...state.mode,
          owner: {
            _tag: "Manager",
            process: command.manager,
            phase: {
              _tag: "RetiringCandidate",
              candidate: state.mode.owner.phase.candidate,
            },
          },
        },
      }
    }
    case "FailSpawning": {
      requireTransition(
        state.mode._tag === "Changing" &&
        state.mode.owner._tag === "Manager" &&
        sameProcess(state.mode.owner.process, command.manager) &&
        state.mode.owner.phase._tag === "Spawning",
        command,
        "manager does not own spawning",
      )
      return {
        revision,
        identityFloor: state.identityFloor,
        mode: {
          _tag: "Unassigned",
          result: Option.some({
            _tag: "Failed",
            changeRevision: state.mode.changeRevision,
            reason: command.reason,
          }),
        },
      }
    }
    case "RetirementBlocked": {
      requireTransition(
        state.mode._tag === "Changing" &&
        state.mode.owner._tag === "Manager" &&
        sameProcess(state.mode.owner.process, command.manager) &&
        state.mode.owner.phase._tag === "RetiringAssigned",
        command,
        "manager does not own predecessor retirement",
      )
      return {
        revision,
        identityFloor: state.identityFloor,
        mode: {
          _tag: "Assigned",
          current: state.mode.owner.phase.current,
          result: Option.some({
            _tag: "Failed",
            changeRevision: state.mode.changeRevision,
            reason: command.reason,
          }),
        },
      }
    }
  }
}

const stateDirectory = (dataDirectory: string): string =>
  NodePath.join(dataDirectory, "acn", "process-state")

const revisionPath = (dataDirectory: string, revision: AcnProcessRevision): string =>
  NodePath.join(stateDirectory(dataDirectory), `${String(revision).padStart(16, "0")}.json`)

const revisionFromFilename = (filename: string): AcnProcessRevision | undefined => {
  const match = /^(\d{16})\.json$/.exec(filename)
  if (match === null) return undefined
  const revision = Number(match[1])
  return Number.isSafeInteger(revision) && revision > 0
    ? AcnProcessRevisionSchema.make(revision)
    : undefined
}

const platformReason = (error: PlatformError): string | undefined =>
  error._tag === "SystemError" ? error.reason : undefined

const stateUnavailable = (path: string, error: unknown): AcnProcessStateUnavailable =>
  new AcnProcessStateUnavailable({ path, reason: String(error) })

export const readAcnProcessStateRevision = (
  dataDirectory: string,
  revision: AcnProcessRevision,
): Effect.Effect<AcnProcessState, AcnProcessStateError, FileSystem.FileSystem> => {
  const path = revisionPath(dataDirectory, revision)
  return Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const text = yield* fs.readFileString(path).pipe(
      Effect.mapError((error) => stateUnavailable(path, error)),
    )
    const state = yield* Schema.decodeUnknown(Schema.parseJson(AcnProcessStateSchema))(text).pipe(
      Effect.mapError((error) => new AcnProcessStateInvalid({ path, reason: String(error) })),
    )
    if (state.revision !== revision) {
      return yield* new AcnProcessStateInvalid({
        path,
        reason: `encoded revision ${state.revision} does not match filename revision ${revision}`,
      })
    }
    return state
  })
}

export const readAcnProcessState = (
  dataDirectory: string,
): Effect.Effect<Option.Option<AcnProcessState>, AcnProcessStateError, FileSystem.FileSystem> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const directory = stateDirectory(dataDirectory)
    const filenames = yield* fs.readDirectory(directory).pipe(
      Effect.catchAll((error) =>
        platformReason(error) === "NotFound"
          ? Effect.succeed([])
          : Effect.fail(stateUnavailable(directory, error)),
      ),
    )
    const revisions = filenames
      .map(revisionFromFilename)
      .filter((value): value is AcnProcessRevision => value !== undefined)
      .sort((left, right) => left - right)
    for (let index = 0; index < revisions.length; index += 1) {
      if (revisions[index] !== index + 1) {
        return yield* new AcnProcessStateInvalid({
          path: directory,
          reason: `process-state revision history is not consecutive at ${index + 1}`,
        })
      }
    }
    const revision = revisions.at(-1)
    if (revision === undefined) return Option.none()
    return Option.some(yield* readAcnProcessStateRevision(dataDirectory, revision))
  })

const sameExpectedRevision = (
  state: Option.Option<AcnProcessState>,
  expected: Option.Option<AcnProcessRevision>,
): boolean => Option.match(state, {
  onNone: () => Option.isNone(expected),
  onSome: (current) => Option.isSome(expected) && expected.value === current.revision,
})

/** Applies one typed command by exclusively publishing the next immutable revision. */
export const applyAcnProcessCommand = (input: {
  readonly dataDirectory: string
  readonly expectedRevision: Option.Option<AcnProcessRevision>
  readonly command: AcnProcessCommand
}): Effect.Effect<AcnProcessState, AcnProcessStateError, FileSystem.FileSystem> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const current = yield* readAcnProcessState(input.dataDirectory)
    if (!sameExpectedRevision(current, input.expectedRevision)) {
      return yield* new AcnProcessStateConflict({
        expected: input.expectedRevision,
        actual: Option.map(current, (state) => state.revision),
      })
    }
    const next = yield* Effect.try({
      try: () => reduceAcnProcessState(current, input.command),
      catch: (error) => error instanceof AcnProcessTransitionRejected
        ? error
        : new AcnProcessTransitionRejected({
            command: input.command._tag,
            reason: String(error),
          }),
    })
    const path = revisionPath(input.dataDirectory, next.revision)
    const directory = stateDirectory(input.dataDirectory)
    yield* fs.makeDirectory(directory, { recursive: true }).pipe(
      Effect.mapError((error) => stateUnavailable(directory, error)),
    )
    const encoded = yield* Schema.encode(Schema.parseJson(AcnProcessStateSchema, { space: 2 }))(next).pipe(
      Effect.mapError((error) => new AcnProcessStateInvalid({ path, reason: String(error) })),
    )
    const temporaryPath = `${path}.${process.pid}.${randomUUID()}.tmp`
    yield* fs.writeFileString(temporaryPath, `${encoded}\n`, { flag: "wx", mode: 0o600 }).pipe(
      Effect.mapError((error) => stateUnavailable(temporaryPath, error)),
    )
    return yield* fs.link(temporaryPath, path).pipe(
      Effect.as(next),
      Effect.catchAll((error) =>
        platformReason(error) === "AlreadyExists"
          ? readAcnProcessState(input.dataDirectory).pipe(
              Effect.flatMap((actual) => Effect.fail(new AcnProcessStateConflict({
                expected: input.expectedRevision,
                actual: Option.map(actual, (state) => state.revision),
              }))),
            )
          : Effect.fail(stateUnavailable(path, error)),
      ),
      Effect.ensuring(fs.remove(temporaryPath, { force: true }).pipe(Effect.ignore)),
    )
  })
