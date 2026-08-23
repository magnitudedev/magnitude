import {
  Clock,
  Data,
  Deferred,
  Duration,
  Effect,
  Exit,
  Option,
  Ref,
  Scope,
} from "effect"
import type { SessionWorkStatus } from "@magnitudedev/agent"

export class SessionRuntimeRetired extends Data.TaggedError("SessionRuntimeRetired")<{
  readonly sessionId: string
  readonly generation: number
}> {}

export interface SessionRuntimeStateSnapshot {
  readonly phase: "open" | "retirement-claimed" | "retired"
  readonly scopedUseCount: number
  readonly scopedUseLabels: ReadonlyArray<string>
  readonly workStatus: SessionWorkStatus
  readonly idleSince: number | null
  readonly revision: number
}

export interface SessionRetirementClaim {
  readonly sessionId: string
  readonly generation: number
  readonly reason: string
  readonly revision: number
}

export interface SessionRuntimeState {
  readonly acquire: (label: string) => Effect.Effect<void, SessionRuntimeRetired, Scope.Scope>
  readonly acquireIfActive: (
    label: string,
  ) => Effect.Effect<Option.Option<void>, SessionRuntimeRetired, Scope.Scope>
  readonly updateWorkStatus: (status: SessionWorkStatus) => Effect.Effect<void>
  readonly retireNow: (reason: string) => Effect.Effect<boolean>
  readonly snapshot: Effect.Effect<SessionRuntimeStateSnapshot>
}

interface OpenState {
  readonly phase: "open"
  readonly accepting: boolean
  readonly scopedUses: ReadonlyMap<number, string>
  readonly workStatus: SessionWorkStatus
  readonly idleSince: number | null
  readonly revision: number
  readonly changed: Deferred.Deferred<void>
}

interface ClaimedState {
  readonly phase: "retirement-claimed"
  readonly claimId: number
  readonly reason: string
  readonly workStatus: SessionWorkStatus
  readonly revision: number
  readonly changed: Deferred.Deferred<void>
  readonly resolution: Deferred.Deferred<"rolled-back" | "retired">
}

interface RetiredState {
  readonly phase: "retired"
  readonly workStatus: SessionWorkStatus
  readonly revision: number
  readonly changed: Deferred.Deferred<void>
}

type RuntimeState = OpenState | ClaimedState | RetiredState

type Admission =
  | { readonly _tag: "acquired"; readonly token: number; readonly notify: Deferred.Deferred<void> }
  | { readonly _tag: "wait"; readonly resolution: Deferred.Deferred<"rolled-back" | "retired"> }
  | { readonly _tag: "inactive" }
  | { readonly _tag: "retired" }

type ClaimAttempt =
  | {
      readonly _tag: "claimed"
      readonly claimId: number
      readonly revision: number
      readonly workStatus: SessionWorkStatus
      readonly resolution: Deferred.Deferred<"rolled-back" | "retired">
      readonly notify: Deferred.Deferred<void>
    }
  | { readonly _tag: "wait"; readonly changed: Deferred.Deferred<void>; readonly delayMs: number | null }
  | { readonly _tag: "done" }

export interface SessionRuntimeStateOptions<E = never, R = never> {
  readonly sessionId: string
  readonly generation: number
  readonly idleTimeout: Duration.DurationInput
  readonly readWorkStatus: Effect.Effect<SessionWorkStatus>
  readonly retire: (claim: SessionRetirementClaim) => Effect.Effect<boolean, E, R>
}

const monotonicMillis = Clock.currentTimeNanos.pipe(
  Effect.map((nanos) => Number(nanos / 1_000_000n)),
)

const completeChange = (changed: Deferred.Deferred<void>) =>
  Deferred.succeed(changed, undefined).pipe(Effect.asVoid)

const sameWorkStatus = (left: SessionWorkStatus, right: SessionWorkStatus) =>
  left._tag === right._tag && left.workerCount === right.workerCount

export const makeSessionRuntimeState = <E = never, R = never>(
  options: SessionRuntimeStateOptions<E, R>,
): Effect.Effect<SessionRuntimeState, never, Scope.Scope | R> =>
  Effect.gen(function* () {
    const idleTimeoutMs = Duration.toMillis(Duration.decode(options.idleTimeout))
    if (!Number.isFinite(idleTimeoutMs) || idleTimeoutMs < 0) {
      return yield* Effect.dieMessage(
        "Session idle timeout must be a finite, non-negative duration",
      )
    }

    const retireContext = yield* Effect.context<R>()
    const initialStatus = yield* options.readWorkStatus
    const startedAt = yield* monotonicMillis
    const initialChanged = yield* Deferred.make<void>()
    const state = yield* Ref.make<RuntimeState>({
      phase: "open",
      accepting: true,
      scopedUses: new Map(),
      workStatus: initialStatus,
      idleSince: initialStatus._tag === "Quiescent" ? startedAt : null,
      revision: 0,
      changed: initialChanged,
    })
    let nextToken = 0
    let nextClaim = 0

    const retiredError = () => new SessionRuntimeRetired({
      sessionId: options.sessionId,
      generation: options.generation,
    })

    const updateWorkStatus = (status: SessionWorkStatus) =>
      Effect.gen(function* () {
        const now = yield* monotonicMillis
        const nextChanged = yield* Deferred.make<void>()
        const notify = yield* Ref.modify(
          state,
          (current): readonly [Deferred.Deferred<void> | null, RuntimeState] => {
            if (current.phase !== "open" || sameWorkStatus(current.workStatus, status)) {
              return [null, current]
            }
            return [
              current.changed,
              {
                ...current,
                workStatus: status,
                idleSince: status._tag === "Quiescent" && current.scopedUses.size === 0
                  ? now
                  : null,
                revision: current.revision + 1,
                changed: nextChanged,
              },
            ]
          },
        )
        if (notify) yield* completeChange(notify)
      }).pipe(Effect.uninterruptible)

    const refreshWorkStatus = options.readWorkStatus.pipe(
      Effect.flatMap(updateWorkStatus),
    )

    const release = (token: number) =>
      Effect.gen(function* () {
        const status = yield* options.readWorkStatus
        const now = yield* monotonicMillis
        const nextChanged = yield* Deferred.make<void>()
        const notify = yield* Ref.modify(
          state,
          (current): readonly [Deferred.Deferred<void> | null, RuntimeState] => {
            if (current.phase !== "open" || !current.scopedUses.has(token)) {
              return [null, current]
            }
            const scopedUses = new Map(current.scopedUses)
            scopedUses.delete(token)
            const changed = !sameWorkStatus(current.workStatus, status) || scopedUses.size === 0
            return [
              changed ? current.changed : null,
              {
                ...current,
                scopedUses,
                workStatus: status,
                idleSince: scopedUses.size === 0 && status._tag === "Quiescent" ? now : null,
                revision: changed ? current.revision + 1 : current.revision,
                changed: changed ? nextChanged : current.changed,
              },
            ]
          },
        )
        if (notify) yield* completeChange(notify)
      }).pipe(Effect.uninterruptible)

    const admit = (
      label: string,
      onlyIfActive: boolean,
    ): Effect.Effect<Option.Option<Effect.Effect<void>>, SessionRuntimeRetired> =>
      Effect.suspend(() =>
        Effect.uninterruptibleMask((restore) =>
          Effect.gen(function* () {
            const changed = yield* Deferred.make<void>()
            const admission = yield* Ref.modify(
              state,
              (current): readonly [Admission, RuntimeState] => {
                if (current.phase === "retired") {
                  return [{ _tag: "retired" }, current]
                }
                if (current.phase === "retirement-claimed") {
                  return [{ _tag: "wait", resolution: current.resolution }, current]
                }
                if (!current.accepting) return [{ _tag: "retired" }, current]
                const active = current.scopedUses.size > 0 || current.workStatus._tag === "Working"
                if (onlyIfActive && !active) return [{ _tag: "inactive" }, current]

                const token = ++nextToken
                return [
                  { _tag: "acquired", token, notify: current.changed },
                  {
                    ...current,
                    scopedUses: new Map(current.scopedUses).set(token, label),
                    idleSince: null,
                    revision: current.revision + 1,
                    changed,
                  },
                ]
              },
            )

            if (admission._tag === "retired") return yield* retiredError()
            if (admission._tag === "inactive") return Option.none()
            if (admission._tag === "wait") {
              const resolution = yield* restore(Deferred.await(admission.resolution))
              if (resolution === "retired") return yield* retiredError()
              return yield* restore(admit(label, onlyIfActive))
            }
            yield* completeChange(admission.notify)
            return Option.some(release(admission.token))
          }),
        ),
      )

    const acquireScoped = (label: string, onlyIfActive: boolean) =>
      Effect.uninterruptibleMask((restore) =>
        restore(admit(label, onlyIfActive)).pipe(
          Effect.tap((releaseOption) =>
            Option.match(releaseOption, {
              onNone: () => Effect.void,
              onSome: (finalize) => Effect.addFinalizer(() => finalize),
            }),
          ),
        ),
      )

    const closeAdmission = Effect.gen(function* () {
      const changed = yield* Deferred.make<void>()
      const notify = yield* Ref.modify(
        state,
        (current): readonly [Deferred.Deferred<void> | null, RuntimeState] => {
          if (current.phase !== "open" || !current.accepting) return [null, current]
          return [
            current.changed,
            {
              ...current,
              accepting: false,
              revision: current.revision + 1,
              changed,
            },
          ]
        },
      )
      if (notify) yield* completeChange(notify)
    }).pipe(Effect.uninterruptible)

    const claim = (reason: string, force: boolean): Effect.Effect<ClaimAttempt> =>
      refreshWorkStatus.pipe(
        Effect.zipRight(
          Effect.gen(function* () {
            const now = yield* monotonicMillis
            const changed = yield* Deferred.make<void>()
            const resolution = yield* Deferred.make<"rolled-back" | "retired">()
            const result = yield* Ref.modify(
              state,
              (current): readonly [ClaimAttempt, RuntimeState] => {
                if (current.phase === "retired") return [{ _tag: "done" }, current]
                if (current.phase === "retirement-claimed") {
                  return [{ _tag: "wait", changed: current.changed, delayMs: null }, current]
                }
                if (current.scopedUses.size > 0 || current.workStatus._tag === "Working") {
                  return [{ _tag: "wait", changed: current.changed, delayMs: null }, current]
                }

                const idleSince = current.idleSince ?? now
                const remaining = idleSince + idleTimeoutMs - now
                if (!force && remaining > 0) {
                  return [{ _tag: "wait", changed: current.changed, delayMs: remaining }, current]
                }

                const claimId = ++nextClaim
                const revision = current.revision + 1
                return [
                  {
                    _tag: "claimed",
                    claimId,
                    revision,
                    workStatus: current.workStatus,
                    resolution,
                    notify: current.changed,
                  },
                  {
                    phase: "retirement-claimed",
                    claimId,
                    reason,
                    workStatus: current.workStatus,
                    revision,
                    changed,
                    resolution,
                  },
                ]
              },
            )
            if (result._tag === "claimed") yield* completeChange(result.notify)
            return result
          }).pipe(Effect.uninterruptible),
        ),
      )

    const resolveClaim = (
      attempt: Extract<ClaimAttempt, { readonly _tag: "claimed" }>,
      commit: boolean,
    ) => Effect.gen(function* () {
      const now = yield* monotonicMillis
      const nextChanged = yield* Deferred.make<void>()
      const result = yield* Ref.modify(
        state,
        (current): readonly [
          { readonly applied: boolean; readonly changed: Deferred.Deferred<void> | null },
          RuntimeState,
        ] => {
          if (current.phase !== "retirement-claimed" || current.claimId !== attempt.claimId) {
            return [{ applied: false, changed: null }, current]
          }
          if (commit) {
            return [
              { applied: true, changed: current.changed },
              {
                phase: "retired",
                workStatus: current.workStatus,
                revision: current.revision + 1,
                changed: nextChanged,
              },
            ]
          }
          return [
            { applied: true, changed: current.changed },
            {
              phase: "open",
              accepting: true,
              scopedUses: new Map(),
              workStatus: current.workStatus,
              idleSince: now,
              revision: current.revision + 1,
              changed: nextChanged,
            },
          ]
        },
      )
      if (!result.applied) return false
      if (result.changed) yield* completeChange(result.changed)
      yield* Deferred.succeed(attempt.resolution, commit ? "retired" : "rolled-back")
      return true
    }).pipe(Effect.uninterruptible)

    const executeClaim = (
      attempt: Extract<ClaimAttempt, { readonly _tag: "claimed" }>,
      reason: string,
    ) => Effect.uninterruptibleMask((restore) =>
      restore(
        options.retire({
          sessionId: options.sessionId,
          generation: options.generation,
          reason,
          revision: attempt.revision,
        }).pipe(Effect.provide(retireContext)),
      ).pipe(
        Effect.exit,
        Effect.flatMap((exit) => {
          const commit = Exit.isSuccess(exit) && exit.value
          return resolveClaim(attempt, commit).pipe(
            Effect.map((applied) => applied && commit),
          )
        }),
      ),
    )

    const waitForAttempt = (attempt: Extract<ClaimAttempt, { readonly _tag: "wait" }>) => {
      const changed = Deferred.await(attempt.changed)
      return attempt.delayMs === null
        ? changed
        : Effect.raceFirst(
            changed,
            Effect.sleep(Duration.millis(Math.max(0, attempt.delayMs))),
          )
    }

    const deadlineLoop: Effect.Effect<void> = Effect.suspend(() =>
      claim("idle-timeout", false).pipe(
        Effect.flatMap((attempt) => {
          if (attempt._tag === "done") return Effect.void
          if (attempt._tag === "wait") {
            return waitForAttempt(attempt).pipe(Effect.zipRight(deadlineLoop))
          }
          return executeClaim(attempt, "idle-timeout").pipe(Effect.zipRight(deadlineLoop))
        }),
      ),
    )
    yield* deadlineLoop.pipe(Effect.forkScoped)

    const retireNow = (reason: string): Effect.Effect<boolean> =>
      closeAdmission.pipe(
        Effect.zipRight(claim(reason, true)),
        Effect.flatMap((attempt) => {
          if (attempt._tag === "done") return Effect.succeed(false)
          if (attempt._tag === "wait") {
            return waitForAttempt(attempt).pipe(Effect.zipRight(retireNow(reason)))
          }
          return executeClaim(attempt, reason)
        }),
      )

    const snapshot = Ref.get(state).pipe(
      Effect.map((current): SessionRuntimeStateSnapshot => ({
        phase: current.phase,
        scopedUseCount: current.phase === "open" ? current.scopedUses.size : 0,
        scopedUseLabels: current.phase === "open" ? [...current.scopedUses.values()] : [],
        workStatus: current.workStatus,
        idleSince: current.phase === "open" ? current.idleSince : null,
        revision: current.revision,
      })),
    )

    return {
      acquire: (label) => acquireScoped(label, false).pipe(
        Effect.flatMap(Option.match({
          onNone: () => Effect.fail(retiredError()),
          onSome: () => Effect.void,
        })),
      ),
      acquireIfActive: (label) => acquireScoped(label, true).pipe(
        Effect.map(Option.map(() => undefined)),
      ),
      updateWorkStatus,
      retireNow,
      snapshot,
    }
  })
