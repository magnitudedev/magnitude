import {
  Context,
  Deferred,
  Duration,
  Effect,
  ExecutionStrategy,
  Exit,
  Layer,
  Option,
  PubSub,
  Ref,
  Scope,
  Stream,
} from "effect"
import { DEFAULT_CHAT_NAME, type SessionWorkStatus } from "@magnitudedev/agent"
import {
  SessionNotFound,
  SessionOperationFailed,
  type SessionError,
} from "@magnitudedev/protocol"
import type { StoredSessionMeta } from "@magnitudedev/storage"
import { AcnActivityTracker } from "./activity-tracker"
import { AgentFactory } from "./agent-factory"
import { makeResourceUseGate, ResourceRetired, type ResourceUseGate } from "./resource-use-gate"
import { SessionStore } from "./session-store"
import {
  SessionRuntimeOptionsStore,
  normalizeSessionRuntimeOptions,
  type SessionRuntimeOptions,
} from "./session-runtime-options"
import type { RuntimeEntry } from "./session-types"

export interface RuntimeStartRequest {
  readonly sessionId: string
  readonly cwd: string
  readonly options: SessionRuntimeOptions
  readonly visibility: StoredSessionMeta["visibility"]
}

export interface ResidentSessionSnapshot {
  readonly sessionId: string
  readonly generation: number
  readonly title: string
  readonly cwd: string
  readonly scratchpadPath: string
  readonly createdAt: number
  readonly updatedAt: number
  readonly residentSince: number
  readonly workStatus: SessionWorkStatus
}

export interface SessionRetirementObserver {
  readonly retire: (input: {
    readonly sessionId: string
    readonly generation: number
  }) => Effect.Effect<void>
}

export interface AgentRuntimeApi {
  readonly withSession: <A, E, R>(
    sessionId: string,
    label: string,
    use: (entry: RuntimeEntry, generation: number) => Effect.Effect<A, E, R>,
  ) => Effect.Effect<A, E | SessionError, R>
  readonly withSessionRequest: <A, E, R>(
    request: RuntimeStartRequest,
    label: string,
    use: (entry: RuntimeEntry, generation: number) => Effect.Effect<A, E, R>,
  ) => Effect.Effect<A, E | SessionError, R>
  readonly tryWithResident: <A, E, R>(
    sessionId: string,
    label: string,
    use: (entry: RuntimeEntry, generation: number) => Effect.Effect<A, E, R>,
  ) => Effect.Effect<Option.Option<A>, E | SessionError, R>
  readonly tryWithBusyResident: <A, E, R>(
    sessionId: string,
    label: string,
    use: (entry: RuntimeEntry, generation: number) => Effect.Effect<A, E, R>,
  ) => Effect.Effect<Option.Option<A>, E | SessionError, R>
  readonly residentSessions: Effect.Effect<ReadonlyArray<ResidentSessionSnapshot>>
  readonly dispose: (sessionId: string) => Effect.Effect<void>
  readonly deleteSession: <R>(
    sessionId: string,
    removeDurableState: Effect.Effect<void, SessionError, R>,
  ) => Effect.Effect<void, SessionError, R>
  readonly registerRetirementObserver: (
    observer: SessionRetirementObserver,
  ) => Effect.Effect<Effect.Effect<void>>
  readonly changes: Stream.Stream<void>
}

export class AgentRuntime extends Context.Tag("AgentRuntime")<AgentRuntime, AgentRuntimeApi>() {}

interface ResidentGeneration {
  readonly generation: number
  readonly entry: RuntimeEntry
  readonly gate: ResourceUseGate
  readonly scope: Scope.CloseableScope
  readonly residentSince: number
  readonly workStatus: Ref.Ref<SessionWorkStatus>
  readonly reconcileWork: Effect.Effect<void>
}

type StartDeferred = Deferred.Deferred<ResidentGeneration, SessionError>

type StartClaim =
  | { readonly _tag: "owner"; readonly deferred: StartDeferred }
  | { readonly _tag: "joiner"; readonly deferred: StartDeferred }

type DeleteDeferred = Deferred.Deferred<void, SessionError>

type DeleteClaim =
  | { readonly _tag: "owner"; readonly deferred: DeleteDeferred }
  | { readonly _tag: "joiner"; readonly deferred: DeleteDeferred }

export interface AgentRuntimeOptions {
  readonly idleTimeout?: Duration.DurationInput
}

export const makeAgentRuntimeLive = (
  options: AgentRuntimeOptions = {},
): Layer.Layer<AgentRuntime, never, AgentFactory | SessionStore | SessionRuntimeOptionsStore> =>
  Layer.scoped(
    AgentRuntime,
    Effect.gen(function* () {
      const factory = yield* AgentFactory
      const store = yield* SessionStore
      const runtimeOptions = yield* SessionRuntimeOptionsStore
      const rootActivity = yield* Effect.serviceOption(AcnActivityTracker)
      const managerScope = yield* Effect.scope
      const entries = yield* Ref.make(new Map<string, ResidentGeneration>())
      const starts = yield* Ref.make(new Map<string, StartDeferred>())
      const deletions = yield* Ref.make(new Map<string, DeleteDeferred>())
      const admissionLock = yield* Effect.makeSemaphore(1)
      const generations = yield* Ref.make(new Map<string, number>())
      const observers = yield* Ref.make(new Set<SessionRetirementObserver>())
      const changes = yield* PubSub.unbounded<void>()

      const publishChange = PubSub.publish(changes, undefined).pipe(Effect.asVoid)

      const nextGeneration = (sessionId: string) =>
        Ref.modify(generations, (current) => {
          const generation = (current.get(sessionId) ?? 0) + 1
          return [generation, new Map(current).set(sessionId, generation)] as const
        })

      const removeExact = (sessionId: string, generation: number) =>
        Ref.modify(entries, (current) => {
          const resident = current.get(sessionId)
          if (!resident || resident.generation !== generation) {
            return [false, current] as const
          }
          const next = new Map(current)
          next.delete(sessionId)
          return [true, next] as const
        })

      let retireGeneration = (_sessionId: string, _generation: number): Effect.Effect<boolean> =>
        Effect.succeed(true)

      const acquireContinuingLease = (
        sessionId: string,
        generation: number,
        gate: ResourceUseGate,
      ): Effect.Effect<Effect.Effect<void>> =>
        Effect.gen(function* () {
          const rootLease = yield* Option.match(rootActivity, {
            onNone: () => Effect.succeed(Option.none<Effect.Effect<void>>()),
            onSome: (activity) =>
              activity.gate
                .joinIfBusy(`session-work:${sessionId}:${generation}`)
                .pipe(
                  Effect.catchTag("ResourceRetired", () =>
                    Effect.die(new Error("ACN retired while session work started")),
                  ),
                ),
          })
          if (Option.isSome(rootActivity) && Option.isNone(rootLease)) {
            return yield* Effect.die(
              new Error(`Session ${sessionId} became working without ACN demand`),
            )
          }

          const sessionLease = yield* gate
            .joinIfBusy(`continuing-work:${sessionId}:${generation}`)
            .pipe(
              Effect.catchTag("ResourceRetired", () =>
                Effect.die(new Error(`Retired session ${sessionId} became working`)),
              ),
            )
          if (Option.isNone(sessionLease)) {
            if (Option.isSome(rootLease)) yield* rootLease.value
            return yield* Effect.die(
              new Error(`Session ${sessionId} became working without admitted session use`),
            )
          }

          return yield* Effect.succeed(
            sessionLease.value.pipe(
              Effect.zipRight(Option.isSome(rootLease) ? rootLease.value : Effect.void),
            ),
          )
        })

      const makeWorkBridge = (
        sessionId: string,
        generation: number,
        entry: RuntimeEntry,
        gate: ResourceUseGate,
        generationScope: Scope.CloseableScope,
      ) =>
        Effect.gen(function* () {
          const workStatus = yield* Ref.make<SessionWorkStatus>({
            _tag: "Quiescent",
            workerCount: 0,
          })
          const continuingRelease = yield* Ref.make<Effect.Effect<void> | null>(null)
          const serialize = yield* Effect.makeSemaphore(1)

          const reconcileWork = serialize.withPermits(1)(
            Effect.gen(function* () {
              const next = yield* entry.session.state.work.get()
              yield* Ref.set(workStatus, next)
              const release = yield* Ref.get(continuingRelease)
              if (next._tag === "Working" && release === null) {
                const acquired = yield* acquireContinuingLease(sessionId, generation, gate)
                yield* Ref.set(continuingRelease, acquired)
                yield* publishChange
                return
              }
              if (next._tag === "Quiescent" && release !== null) {
                yield* Ref.set(continuingRelease, null)
                yield* release
                yield* publishChange
              }
            }),
          )

          yield* reconcileWork
          yield* Effect.forkIn(
            entry.session.state.work.subscribe.pipe(
              Stream.runForEach(() => reconcileWork),
              Effect.ensuring(
                Ref.getAndSet(continuingRelease, null).pipe(
                  Effect.flatMap((release) => release ?? Effect.void),
                ),
              ),
            ),
            generationScope,
          )
          return { workStatus, reconcileWork }
        })

      const startResident = Effect.fn("acn.agent-runtime.start")(function* (
        request: RuntimeStartRequest,
      ) {
        const generation = yield* nextGeneration(request.sessionId)
        const generationScope = yield* Scope.fork(managerScope, ExecutionStrategy.sequential)
        const gate = yield* makeResourceUseGate({
          resource: `session:${request.sessionId}`,
          generation,
          idleTimeout: options.idleTimeout ?? "2 minutes",
          retire: () => retireGeneration(request.sessionId, generation),
        }).pipe(Effect.provideService(Scope.Scope, managerScope))
        const releaseStartup = yield* gate.acquire("session-start").pipe(Effect.orDie)

        return yield* Effect.gen(function* () {
          const requestedCwd = yield* store.validateCwd(request.cwd)
          yield* runtimeOptions.write(request.sessionId, request.options)
          const session = yield* factory.createSession({
            sessionId: request.sessionId,
            cwd: requestedCwd,
            scope: generationScope,
            options: request.options,
            visibility: request.visibility,
          })
          const residentSince = Date.now()
          const storedMeta = yield* store.readMeta(request.sessionId)
          const createdAt = storedMeta
            ? Date.parse(storedMeta.created) || residentSince
            : residentSince
          const scratchpadPath = yield* store.getScratchpadPath(request.sessionId)
          const entry: RuntimeEntry = {
            id: request.sessionId,
            createdAt,
            updatedAt: residentSince,
            title: storedMeta?.chatName ?? DEFAULT_CHAT_NAME,
            cwd: requestedCwd,
            scratchpadPath,
            session,
            scope: generationScope,
          }
          const bridge = yield* makeWorkBridge(
            request.sessionId,
            generation,
            entry,
            gate,
            generationScope,
          )
          const resident: ResidentGeneration = {
            generation,
            entry,
            gate,
            scope: generationScope,
            residentSince,
            ...bridge,
          }
          yield* Ref.update(entries, (current) => new Map(current).set(request.sessionId, resident))
          yield* publishChange
          return { resident, releaseStartup }
        }).pipe(
          Effect.onExit((exit) =>
            Exit.isSuccess(exit)
              ? Effect.void
              : releaseStartup.pipe(Effect.zipRight(Scope.close(generationScope, Exit.void))),
          ),
        )
      })

      const claimStart = (sessionId: string) =>
        Effect.gen(function* () {
          const deferred = yield* Deferred.make<ResidentGeneration, SessionError>()
          return yield* Ref.modify(
            starts,
            (current): readonly [StartClaim, Map<string, StartDeferred>] => {
              const existing = current.get(sessionId)
              if (existing) return [{ _tag: "joiner", deferred: existing }, current]
              return [{ _tag: "owner", deferred }, new Map(current).set(sessionId, deferred)]
            },
          )
        })

      const clearStart = (sessionId: string, deferred: StartDeferred) =>
        Ref.update(starts, (current) => {
          if (current.get(sessionId) !== deferred) return current
          const next = new Map(current)
          next.delete(sessionId)
          return next
        })

      const requestForStoredSession = Effect.fn("acn.agent-runtime.request-for-stored-session")(
        function* (sessionId: string) {
          const meta = yield* store.readMeta(sessionId)
          if (!meta) return yield* new SessionNotFound({ sessionId })
          return {
            sessionId,
            cwd: meta.workingDirectory,
            options: (yield* runtimeOptions.read(sessionId)) ?? normalizeSessionRuntimeOptions(),
            visibility: meta.visibility,
          } satisfies RuntimeStartRequest
        },
      )

      /** Resolve a generation and acquire the caller lease atomically. */
      const acquireResident = (
        request: RuntimeStartRequest,
        label: string,
      ): Effect.Effect<
        readonly [ResidentGeneration, Effect.Effect<void>],
        SessionError | ResourceRetired
      > =>
        Effect.uninterruptibleMask((restore) =>
          Effect.gen(function* () {
            const deleting = () =>
              new SessionOperationFailed({
                operation: `session ${request.sessionId}`,
                reason: "Session is being deleted",
              })
            const resolved = yield* restore(
              admissionLock.withPermits(1)(
                Effect.gen(function* () {
                  if ((yield* Ref.get(deletions)).has(request.sessionId)) {
                    return yield* deleting()
                  }
                  const existing = (yield* Ref.get(entries)).get(request.sessionId)
                  if (existing) {
                    const release = yield* existing.gate.acquire(label)
                    return { _tag: "resident" as const, resident: existing, release }
                  }
                  return { _tag: "start" as const, claim: yield* claimStart(request.sessionId) }
                }),
              ),
            )
            if (resolved._tag === "resident") {
              return [resolved.resident, resolved.release] as const
            }

            const claim = resolved.claim
            if (claim._tag === "joiner") {
              const resident = yield* restore(Deferred.await(claim.deferred))
              const release = yield* restore(
                admissionLock.withPermits(1)(
                  Effect.gen(function* () {
                    if ((yield* Ref.get(deletions)).has(request.sessionId)) {
                      return yield* deleting()
                    }
                    return yield* resident.gate.acquire(label)
                  }),
                ),
              )
              return [resident, release] as const
            }

            const result = yield* restore(startResident(request)).pipe(Effect.exit)
            if (Exit.isFailure(result)) {
              yield* Deferred.failCause(claim.deferred, result.cause)
              yield* clearStart(request.sessionId, claim.deferred)
              return yield* Effect.failCause(result.cause)
            }

            const { resident, releaseStartup } = result.value
            const release = yield* resident.gate.acquire(label).pipe(Effect.orDie)
            yield* Deferred.succeed(claim.deferred, resident)
            yield* clearStart(request.sessionId, claim.deferred)
            yield* releaseStartup
            return [resident, release] as const
          }),
        )

      const useResident = <A, E, R>(
        resident: ResidentGeneration,
        label: string,
        use: (entry: RuntimeEntry, generation: number) => Effect.Effect<A, E, R>,
      ): Effect.Effect<A, E | ResourceRetired, R> =>
        resident.gate.withUse(
          label,
          use(resident.entry, resident.generation).pipe(Effect.ensuring(resident.reconcileWork)),
        )

      const withRequest = <A, E, R>(
        request: RuntimeStartRequest,
        label: string,
        use: (entry: RuntimeEntry, generation: number) => Effect.Effect<A, E, R>,
      ): Effect.Effect<A, E | SessionError, R> =>
        Effect.suspend(() =>
          acquireResident(request, label).pipe(
            Effect.flatMap(([resident, release]) =>
              use(resident.entry, resident.generation).pipe(
                Effect.ensuring(resident.reconcileWork),
                Effect.ensuring(release),
              ),
            ),
            Effect.catchTag("ResourceRetired", () => withRequest(request, label, use)),
          ),
        )

      const withSession = <A, E, R>(
        sessionId: string,
        label: string,
        use: (entry: RuntimeEntry, generation: number) => Effect.Effect<A, E, R>,
      ): Effect.Effect<A, E | SessionError, R> =>
        requestForStoredSession(sessionId).pipe(
          Effect.flatMap((request) => withRequest(request, label, use)),
        )

      const tryWithResident = <A, E, R>(
        sessionId: string,
        label: string,
        use: (entry: RuntimeEntry, generation: number) => Effect.Effect<A, E, R>,
      ): Effect.Effect<Option.Option<A>, E | SessionError, R> =>
        Effect.suspend(() =>
          Effect.gen(function* () {
            const resident = (yield* Ref.get(entries)).get(sessionId)
            if (!resident) return Option.none<A>()
            return yield* useResident(resident, label, use).pipe(
              Effect.map(Option.some),
              Effect.catchTag("ResourceRetired", () => tryWithResident(sessionId, label, use)),
            )
          }),
        )

      const tryWithBusyResident = <A, E, R>(
        sessionId: string,
        label: string,
        use: (entry: RuntimeEntry, generation: number) => Effect.Effect<A, E, R>,
      ): Effect.Effect<Option.Option<A>, E | SessionError, R> =>
        Effect.suspend(() =>
          Effect.gen(function* () {
            const resident = (yield* Ref.get(entries)).get(sessionId)
            if (!resident) return Option.none<A>()
            return yield* resident.gate
              .withBusyUse(label, use(resident.entry, resident.generation))
              .pipe(
                Effect.catchTag("ResourceRetired", () =>
                  tryWithBusyResident(sessionId, label, use),
                ),
              )
          }),
        )

      retireGeneration = (sessionId, generation) =>
        Effect.gen(function* () {
          const resident = (yield* Ref.get(entries)).get(sessionId)
          if (!resident || resident.generation !== generation) return true

          for (const observer of yield* Ref.get(observers)) {
            yield* observer.retire({ sessionId, generation }).pipe(
              Effect.catchAllCause((cause) =>
                Effect.logWarning("Session retirement observer failed").pipe(
                  Effect.annotateLogs({
                    sessionId,
                    generation,
                    cause: String(cause),
                  }),
                ),
              ),
            )
          }
          yield* Scope.close(resident.scope, Exit.void)
          const removed = yield* removeExact(sessionId, generation)
          if (removed) {
            yield* publishChange
            yield* Effect.logInfo("Evicted idle session").pipe(
              Effect.annotateLogs({ sessionId, generation }),
            )
          }
          return true
        })

      const dispose = Effect.fn("acn.agent-runtime.dispose")(function* (sessionId: string) {
        const resident = (yield* Ref.get(entries)).get(sessionId)
        if (!resident) return
        yield* resident.gate.retireNow("explicit-dispose")
      })

      const deleteSession: AgentRuntimeApi["deleteSession"] = (sessionId, removeDurableState) =>
        Effect.uninterruptible(
          Effect.gen(function* () {
            const candidate = yield* Deferred.make<void, SessionError>()
            const claim = yield* admissionLock.withPermits(1)(
              Ref.modify(
                deletions,
                (current): readonly [DeleteClaim, Map<string, DeleteDeferred>] => {
                const existing = current.get(sessionId)
                  if (existing) return [{ _tag: "joiner", deferred: existing }, current]
                return [
                    { _tag: "owner", deferred: candidate },
                  new Map(current).set(sessionId, candidate),
                ] as const
                },
              ),
            )
            if (claim._tag === "joiner") return yield* Deferred.await(claim.deferred)

            const result = yield* Effect.gen(function* () {
              const pendingStart = (yield* Ref.get(starts)).get(sessionId)
              if (pendingStart) yield* Deferred.await(pendingStart).pipe(Effect.exit)
              yield* dispose(sessionId)
              yield* removeDurableState
            }).pipe(Effect.exit)
            yield* Deferred.done(candidate, result)
            yield* admissionLock.withPermits(1)(
              Ref.update(deletions, (current) => {
                if (current.get(sessionId) !== candidate) return current
                const next = new Map(current)
                next.delete(sessionId)
                return next
              }),
            )
            return yield* Deferred.await(candidate)
          }),
        )
      return {
        withSession,
        withSessionRequest: withRequest,
        tryWithResident,
        tryWithBusyResident,
        residentSessions: Effect.gen(function* () {
          const result: ResidentSessionSnapshot[] = []
          for (const resident of (yield* Ref.get(entries)).values()) {
            result.push({
              sessionId: resident.entry.id,
              generation: resident.generation,
              title: resident.entry.title,
              cwd: resident.entry.cwd,
              scratchpadPath: resident.entry.scratchpadPath,
              createdAt: resident.entry.createdAt,
              updatedAt: resident.entry.updatedAt,
              residentSince: resident.residentSince,
              workStatus: yield* Ref.get(resident.workStatus),
            })
          }
          return result
        }),
        dispose,
        deleteSession,
        registerRetirementObserver: (observer) =>
          Ref.update(observers, (current) => new Set(current).add(observer)).pipe(
            Effect.as(
              Ref.update(observers, (current) => {
                const next = new Set(current)
                next.delete(observer)
                return next
              }),
            ),
          ),
        changes: Stream.fromPubSub(changes).pipe(Stream.map(() => undefined)),
      } satisfies AgentRuntimeApi
    }),
  )

export const AgentRuntimeLive = makeAgentRuntimeLive()
