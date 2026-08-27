import {
  Cause,
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
  InvalidSessionPath,
  SessionNotFound,
  SessionOperationFailed,
  type DirectoryPath,
  type SessionError,
} from "@magnitudedev/acn-protocol"
import { MagnitudeStorage, type StoredSessionMeta } from "@magnitudedev/storage"
import { AgentFactory } from "./agent-factory"
import { FileSystemManager } from "./file-system-manager"
import {
  makeSessionRuntimeState,
  type SessionRuntimeRetired,
  type SessionRuntimeState,
} from "./session-runtime-state"
import {
  SessionRuntimeOptionsStore,
  normalizeSessionRuntimeOptions,
  type SessionRuntimeOptions,
} from "./session-runtime-options"
import type { RuntimeEntry } from "./session-types"

export interface RuntimeStartRequest {
  readonly sessionId: string
  readonly cwd: DirectoryPath
  readonly options: SessionRuntimeOptions
  readonly visibility: StoredSessionMeta["visibility"]
}

export interface SessionRuntimeSnapshot {
  readonly sessionId: string
  readonly generation: number
  readonly title: string
  readonly cwd: DirectoryPath
  readonly scratchpadPath: string
  readonly createdAt: number
  readonly updatedAt: number
  readonly loadedAt: number
  readonly workStatus: SessionWorkStatus
  readonly phase: "open" | "retirement-claimed" | "retired"
  readonly scopedUseCount: number
  readonly scopedUseLabels: ReadonlyArray<string>
  readonly idleSince: number | null
  readonly revision: number
  readonly retirement: SessionRetirementSnapshot | null
}

export interface SessionRuntime {
  readonly entry: RuntimeEntry
  readonly generation: number
}

export type SessionRetirementStage =
  | "notifying-observers"
  | "closing-runtime"
  | "removing-generation"

export interface SessionRetirementSnapshot {
  readonly stage: SessionRetirementStage
  readonly startedAt: number
  readonly stageStartedAt: number
}

export interface SessionRetirementObserver {
  readonly retire: (input: {
    readonly sessionId: string
    readonly generation: number
  }) => Effect.Effect<void>
}

export interface AgentRuntimeApi {
  readonly acquireSession: (
    sessionId: string,
    label: string,
  ) => Effect.Effect<SessionRuntime, SessionError, Scope.Scope>
  readonly acquireSessionRequest: (
    request: RuntimeStartRequest,
    label: string,
  ) => Effect.Effect<SessionRuntime, SessionError, Scope.Scope>
  readonly tryAcquireActiveSession: (
    sessionId: string,
    label: string,
  ) => Effect.Effect<Option.Option<SessionRuntime>, SessionError, Scope.Scope>
  readonly sessionRuntimes: Effect.Effect<ReadonlyArray<SessionRuntimeSnapshot>>
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

interface SessionRuntimeInternal {
  readonly generation: number
  readonly entry: RuntimeEntry
  readonly state: SessionRuntimeState
  readonly scope: Scope.CloseableScope
  readonly loadedAt: number
}

type StartDeferred = Deferred.Deferred<SessionRuntimeInternal, SessionError>

type StartClaim =
  | { readonly _tag: "owner"; readonly deferred: StartDeferred }
  | { readonly _tag: "joiner"; readonly deferred: StartDeferred }

type DeleteDeferred = Deferred.Deferred<void, SessionError>

type DeleteClaim =
  | { readonly _tag: "owner"; readonly deferred: DeleteDeferred }
  | { readonly _tag: "joiner"; readonly deferred: DeleteDeferred }

export interface AgentRuntimeOptions {
  readonly idleTimeout?: Duration.DurationInput
  readonly retirementAdmissionTimeout?: Duration.DurationInput
  readonly retirementShutdownTimeout?: Duration.DurationInput
}

export const makeAgentRuntimeLive = (
  options: AgentRuntimeOptions = {},
): Layer.Layer<
  AgentRuntime,
  never,
  AgentFactory | MagnitudeStorage | SessionRuntimeOptionsStore | FileSystemManager
> =>
  Layer.scoped(
    AgentRuntime,
    Effect.gen(function* () {
      const factory = yield* AgentFactory
      const storage = yield* MagnitudeStorage
      const fileSystem = yield* FileSystemManager
      const runtimeOptions = yield* SessionRuntimeOptionsStore
      const managerScope = yield* Effect.scope
      const entries = yield* Ref.make(new Map<string, SessionRuntimeInternal>())
      const starts = yield* Ref.make(new Map<string, StartDeferred>())
      const deletions = yield* Ref.make(new Map<string, DeleteDeferred>())
      const admissionLock = yield* Effect.makeSemaphore(1)
      const generations = yield* Ref.make(new Map<string, number>())
      const observers = yield* Ref.make(new Set<SessionRetirementObserver>())
      const retirements = yield* Ref.make(new Map<string, SessionRetirementSnapshot>())
      const changes = yield* PubSub.unbounded<void>()

      const publishChange = PubSub.publish(changes, undefined).pipe(Effect.asVoid)

      const readStoredMeta = (
        sessionId: string,
      ): Effect.Effect<StoredSessionMeta | null, SessionOperationFailed> =>
        storage.sessions.readMeta(sessionId).pipe(
          Effect.mapError((error) => new SessionOperationFailed({
            operation: `read session metadata ${sessionId}`,
            reason: error._tag,
          })),
        )

      const nextGeneration = (sessionId: string) =>
        Ref.modify(generations, (current) => {
          const generation = (current.get(sessionId) ?? 0) + 1
          return [generation, new Map(current).set(sessionId, generation)] as const
        })

      const removeExact = (sessionId: string, generation: number) =>
        Ref.modify(entries, (current) => {
          const runtime = current.get(sessionId)
          if (!runtime || runtime.generation !== generation) {
            return [false, current] as const
          }
          const next = new Map(current)
          next.delete(sessionId)
          return [true, next] as const
        })

      let retireGeneration = (_sessionId: string, _generation: number): Effect.Effect<boolean> =>
        Effect.succeed(true)

      const startRuntimeAttempt = Effect.fn("acn.agent-runtime.start-attempt")(function* (
        request: RuntimeStartRequest,
      ) {
        const generation = yield* nextGeneration(request.sessionId)
        const generationScope = yield* Scope.fork(managerScope, ExecutionStrategy.sequential)

        return yield* Effect.gen(function* () {
          // A request's cwd is immutable session identity: validate the host
          // directory once, then start.
          yield* fileSystem.openDirectory(request.cwd).pipe(
            Effect.mapError(() => new InvalidSessionPath({ path: request.cwd })),
          )
          yield* runtimeOptions.write(request.sessionId, request.options)
          const session = yield* factory.createSession({
            sessionId: request.sessionId,
            cwd: request.cwd,
            scope: generationScope,
            options: request.options,
            visibility: request.visibility,
          })
          const loadedAt = Date.now()
          const storedMeta = yield* readStoredMeta(request.sessionId)
          const createdAt = storedMeta
            ? Date.parse(storedMeta.created) || loadedAt
            : loadedAt
          const scratchpadPath = storage.sessions.paths.sessionScratchpad(request.sessionId)
          const entry: RuntimeEntry = {
            id: request.sessionId,
            createdAt,
            updatedAt: loadedAt,
            title: storedMeta?.chatName ?? DEFAULT_CHAT_NAME,
            cwd: request.cwd,
            scratchpadPath,
            session,
            scope: generationScope,
          }
          const state = yield* makeSessionRuntimeState({
            sessionId: request.sessionId,
            generation,
            idleTimeout: options.idleTimeout ?? "2 minutes",
            readWorkStatus: entry.session.state.work.get(),
            retire: () => retireGeneration(request.sessionId, generation),
          }).pipe(Effect.provideService(Scope.Scope, managerScope))

          const firstStatus = yield* Deferred.make<void>()
          yield* Effect.forkIn(
            entry.session.state.work.subscribe.pipe(
              Stream.runForEach((status) =>
                state.updateWorkStatus(status).pipe(
                  Effect.zipRight(Deferred.succeed(firstStatus, undefined)),
                  Effect.zipRight(publishChange),
                ),
              ),
              Effect.catchAllCause((cause) =>
                Cause.isInterruptedOnly(cause)
                  ? Effect.void
                  : Effect.logError("Session work-status observation failed").pipe(
                    Effect.annotateLogs({
                      sessionId: request.sessionId,
                      generation,
                      cause: String(cause),
                    }),
                  ),
              ),
            ),
            generationScope,
          )
          yield* Deferred.await(firstStatus)
          const runtime: SessionRuntimeInternal = {
            generation,
            entry,
            state,
            scope: generationScope,
            loadedAt,
          }
          yield* Ref.update(entries, (current) => new Map(current).set(request.sessionId, runtime))
          yield* publishChange
          return runtime
        }).pipe(
          Effect.onExit((exit) =>
            Exit.isSuccess(exit)
              ? Effect.void
              : Scope.close(generationScope, Exit.void),
          ),
        )
      })

      const startRuntime = startRuntimeAttempt

      const claimStart = (sessionId: string) =>
        Effect.gen(function* () {
          const deferred = yield* Deferred.make<SessionRuntimeInternal, SessionError>()
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
          const meta = yield* readStoredMeta(sessionId)
          if (!meta) return yield* new SessionNotFound({ sessionId })
          return {
            sessionId,
            cwd: meta.workingDirectory,
            options: (yield* runtimeOptions.read(sessionId)) ?? normalizeSessionRuntimeOptions(),
            visibility: meta.visibility,
          } satisfies RuntimeStartRequest
        },
      )

      const publicRuntime = (runtime: SessionRuntimeInternal): SessionRuntime => ({
        entry: runtime.entry,
        generation: runtime.generation,
      })

      /** Resolve a generation and acquire exact scoped access atomically. */
      const acquireRuntime = (
        request: RuntimeStartRequest,
        label: string,
      ): Effect.Effect<SessionRuntime, SessionError | SessionRuntimeRetired, Scope.Scope> =>
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
                  if (existing) return { _tag: "runtime" as const, runtime: existing }
                  return { _tag: "start" as const, claim: yield* claimStart(request.sessionId) }
                }),
              ),
            )
            if (resolved._tag === "runtime") {
              yield* restore(
                resolved.runtime.state.acquire(label).pipe(
                  Effect.timeoutFail({
                    duration: options.retirementAdmissionTimeout ?? "10 seconds",
                    onTimeout: () =>
                      new SessionOperationFailed({
                        operation: `session ${request.sessionId}`,
                        reason:
                          "The previous idle session runtime did not finish shutting down in time",
                      }),
                  }),
                ),
              )
              return publicRuntime(resolved.runtime)
            }

            const claim = resolved.claim
            if (claim._tag === "joiner") {
              const runtime = yield* restore(Deferred.await(claim.deferred))
              if ((yield* Ref.get(deletions)).has(request.sessionId)) {
                return yield* deleting()
              }
              yield* restore(
                runtime.state.acquire(label).pipe(
                  Effect.timeoutFail({
                    duration: options.retirementAdmissionTimeout ?? "10 seconds",
                    onTimeout: () =>
                      new SessionOperationFailed({
                        operation: `session ${request.sessionId}`,
                        reason:
                          "The newly started session runtime did not become available in time",
                      }),
                  }),
                ),
              )
              return publicRuntime(runtime)
            }

            const result = yield* restore(startRuntime(request)).pipe(Effect.exit)
            if (Exit.isFailure(result)) {
              yield* Deferred.failCause(claim.deferred, result.cause)
              yield* clearStart(request.sessionId, claim.deferred)
              return yield* Effect.failCause(result.cause)
            }

            const runtime = result.value
            yield* Deferred.succeed(claim.deferred, runtime)
            yield* clearStart(request.sessionId, claim.deferred)
            yield* runtime.state.acquire(label)
            return publicRuntime(runtime)
          }),
        )

      const acquireSessionRequest = (
        request: RuntimeStartRequest,
        label: string,
      ): Effect.Effect<SessionRuntime, SessionError, Scope.Scope> =>
        Effect.suspend(() =>
          acquireRuntime(request, label).pipe(
            Effect.catchTag("SessionRuntimeRetired", () =>
              acquireSessionRequest(request, label)),
          ),
        )

      const acquireSession = (
        sessionId: string,
        label: string,
      ): Effect.Effect<SessionRuntime, SessionError, Scope.Scope> =>
        requestForStoredSession(sessionId).pipe(
          Effect.flatMap((request) => acquireSessionRequest(request, label)),
        )

      const tryAcquireActiveSession = (
        sessionId: string,
        label: string,
      ): Effect.Effect<Option.Option<SessionRuntime>, SessionError, Scope.Scope> =>
        Effect.suspend(() =>
          Effect.gen(function* () {
            const runtime = (yield* Ref.get(entries)).get(sessionId)
            if (!runtime) return Option.none<SessionRuntime>()
            return yield* runtime.state.acquireIfActive(label).pipe(
              Effect.map(Option.map(() => publicRuntime(runtime))),
              Effect.catchTag("SessionRuntimeRetired", () =>
                tryAcquireActiveSession(sessionId, label)),
              )
          }),
        )

      retireGeneration = (sessionId, generation) => {
        const retire = Effect.gen(function* () {
          const runtime = (yield* Ref.get(entries)).get(sessionId)
          if (!runtime || runtime.generation !== generation) return true

          const startedAt = Date.now()
          const setRetirementStage = (stage: SessionRetirementStage) =>
            Ref.update(retirements, (current) =>
              new Map(current).set(sessionId, {
                stage,
                startedAt,
                stageStartedAt: Date.now(),
              }),
            )
          const logStage = (stage: SessionRetirementStage, stageStartedAt: number) =>
            Effect.logDebug("Session retirement stage completed").pipe(
              Effect.annotateLogs({
                sessionId,
                generation,
                stage,
                durationMs: Date.now() - stageStartedAt,
              }),
            )

          let stageStartedAt = Date.now()
          yield* setRetirementStage("notifying-observers")
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
          yield* logStage("notifying-observers", stageStartedAt)

          stageStartedAt = Date.now()
          yield* setRetirementStage("closing-runtime")
          const closeExit = yield* Scope.close(runtime.scope, Exit.void).pipe(Effect.exit)
          if (Exit.isFailure(closeExit)) {
            yield* Effect.logError("Session runtime scope failed to close").pipe(
              Effect.annotateLogs({
                sessionId,
                generation,
                cause: String(closeExit.cause),
              }),
            )
            // Keep the exact generation quarantined. A partially finalized
            // session is unsafe to reopen, but it cannot own the ACN process.
            return yield* Effect.never
          }
          yield* logStage("closing-runtime", stageStartedAt)

          stageStartedAt = Date.now()
          yield* setRetirementStage("removing-generation")
          const removed = yield* removeExact(sessionId, generation)
          yield* Ref.update(retirements, (current) => {
            if (!current.has(sessionId)) return current
            const next = new Map(current)
            next.delete(sessionId)
            return next
          })
          yield* logStage("removing-generation", stageStartedAt)
          if (removed) {
            yield* publishChange
            yield* Effect.logInfo("Unloaded idle session runtime").pipe(
              Effect.annotateLogs({ sessionId, generation }),
            )
          }
          return true
        })
        return Effect.gen(function* () {
          const completed = yield* Ref.make(false)
          yield* Effect.sleep(options.retirementShutdownTimeout ?? "15 seconds").pipe(
            Effect.zipRight(
              Effect.gen(function* () {
                if (yield* Ref.get(completed)) return
                const current = (yield* Ref.get(entries)).get(sessionId)
                if (!current || current.generation !== generation) return
                yield* Effect.logError("Session retirement exceeded its liveness deadline").pipe(
                  Effect.annotateLogs({ sessionId, generation }),
                )
              }),
            ),
            Effect.interruptible,
            Effect.forkIn(managerScope),
          )
          return yield* retire.pipe(
            Effect.ensuring(Ref.set(completed, true)),
          )
        })
      }

      const dispose = Effect.fn("acn.agent-runtime.dispose")(function* (sessionId: string) {
        const runtime = (yield* Ref.get(entries)).get(sessionId)
        if (!runtime) return
        yield* runtime.state.retireNow("explicit-dispose")
      })

      const deleteSession: AgentRuntimeApi["deleteSession"] = (sessionId, removeDurableState) =>
        Effect.uninterruptibleMask((restore) =>
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
            if (claim._tag === "joiner") return yield* restore(Deferred.await(claim.deferred))

            const operation = Effect.gen(function* () {
              const result = yield* Effect.gen(function* () {
                const pendingStart = (yield* Ref.get(starts)).get(sessionId)
                if (pendingStart) yield* Deferred.await(pendingStart).pipe(Effect.exit)
                yield* dispose(sessionId)
                yield* removeDurableState
              }).pipe(Effect.exit)
              yield* admissionLock.withPermits(1)(
                Ref.update(deletions, (current) => {
                  if (current.get(sessionId) !== candidate) return current
                  const next = new Map(current)
                  next.delete(sessionId)
                  return next
                }),
              )
              yield* Deferred.done(candidate, result)
            })
            yield* operation.pipe(
              Effect.forkIn(managerScope),
            )
            return yield* restore(Deferred.await(candidate))
          }),
        )
      return {
        acquireSession,
        acquireSessionRequest,
        tryAcquireActiveSession,
        sessionRuntimes: Effect.gen(function* () {
          const result: SessionRuntimeSnapshot[] = []
          for (const runtime of (yield* Ref.get(entries)).values()) {
            const state = yield* runtime.state.snapshot
            result.push({
              sessionId: runtime.entry.id,
              generation: runtime.generation,
              title: runtime.entry.title,
              cwd: runtime.entry.cwd,
              scratchpadPath: runtime.entry.scratchpadPath,
              createdAt: runtime.entry.createdAt,
              updatedAt: runtime.entry.updatedAt,
              loadedAt: runtime.loadedAt,
              workStatus: state.workStatus,
              phase: state.phase,
              scopedUseCount: state.scopedUseCount,
              scopedUseLabels: state.scopedUseLabels,
              idleSince: state.idleSince,
              revision: state.revision,
              retirement: (yield* Ref.get(retirements)).get(runtime.entry.id) ?? null,
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
