import { Cause, Clock, Context, Effect, Either, Exit, Layer, Option } from "effect"
import {
  SessionAlreadyExists,
  SessionMetadataUnreadable,
  SessionMetadataWriteFailed,
  SessionNotArchived,
  SessionNotFound,
  SessionOperationFailed,
  SessionStartFailed,
  type CreateSessionInitial,
  type CreateSessionResult,
  type SessionError,
  type SessionMetadata as ProtocolSessionMetadata,
  type SessionOptions,
} from "@magnitudedev/acn-protocol"
import { MagnitudeStorage, type StoredSessionMeta } from "@magnitudedev/storage"
import { AgentRuntime } from "./agent-runtime"
import { SessionDrafts } from "./session-drafts"
import { SessionCommands } from "./session-commands"
import { sessionErrorMessage } from "./session-errors"
import { decodeSessionMetadata, SessionInspector } from "./session-inspector"
import { hasUserMessageContent, type SessionExecutionContext } from "./session-types"

export type SessionCommandError =
  | SessionNotFound
  | SessionMetadataUnreadable
  | SessionMetadataWriteFailed

export interface SessionLifecycleApi {
  readonly createSession: (
    cwd?: string,
    sessionId?: string,
    initial?: CreateSessionInitial,
    options?: SessionOptions,
    draftOwnerId?: string | null,
  ) => Effect.Effect<CreateSessionResult, SessionError>
  readonly preloadSession: (
    cwd: string,
    options?: SessionOptions,
    draftOwnerId?: string | null,
  ) => Effect.Effect<{ readonly sessionId: string }, SessionError>
  readonly releaseSessionPreload: (
    cwd: string,
    sessionId: string,
    options?: SessionOptions,
    draftOwnerId?: string | null,
  ) => Effect.Effect<void, SessionError>
  readonly archiveSession: (
    sessionId: string,
  ) => Effect.Effect<ProtocolSessionMetadata, SessionCommandError>
  readonly restoreSession: (
    sessionId: string,
  ) => Effect.Effect<ProtocolSessionMetadata, SessionCommandError>
  readonly setSessionPinned: (
    sessionId: string,
    pinned: boolean,
  ) => Effect.Effect<ProtocolSessionMetadata, SessionCommandError>
  readonly deleteArchivedSession: (
    sessionId: string,
  ) => Effect.Effect<void, SessionCommandError | SessionNotArchived>
  readonly getSessionExecutionContext: (
    sessionId: string,
  ) => Effect.Effect<SessionExecutionContext, SessionError>
  readonly getSessionCwd: (sessionId: string) => Effect.Effect<string, SessionError>
}

export class SessionLifecycle extends Context.Tag("SessionLifecycle")<
  SessionLifecycle,
  SessionLifecycleApi
>() {}

export const SessionLifecycleLive: Layer.Layer<
  SessionLifecycle,
  never,
  AgentRuntime | SessionCommands | SessionDrafts | MagnitudeStorage | SessionInspector
> =
  Layer.effect(
    SessionLifecycle,
    Effect.gen(function* () {
      const runtime = yield* AgentRuntime
      const commands = yield* SessionCommands
      const drafts = yield* SessionDrafts
      const storage = yield* MagnitudeStorage
      const inspector = yield* SessionInspector

      const runtimeSnapshot = (sessionId: string) =>
        runtime.sessionRuntimes.pipe(
          Effect.map((sessions) => sessions.find((session) => session.sessionId === sessionId)),
        )

      /** Visible stored metadata for the create-path existence checks. */
      const readExisting = (
        sessionId: string,
      ): Effect.Effect<ProtocolSessionMetadata | null, SessionStartFailed> =>
        inspector.get(sessionId).pipe(
          Effect.map((metadata): ProtocolSessionMetadata | null => metadata),
          Effect.catchTags({
            SessionNotFound: () => Effect.succeed(null),
            SessionMetadataUnreadable: () => Effect.fail(new SessionStartFailed({
              sessionId,
              reason: "Session metadata is unreadable",
            })),
          }),
        )

      const readMetaForCommand = (
        sessionId: string,
      ): Effect.Effect<StoredSessionMeta | null, SessionMetadataUnreadable> =>
        storage.sessions.readMeta(sessionId).pipe(
          Effect.mapError(() => new SessionMetadataUnreadable({ sessionId })),
          Effect.map((meta) => meta?.visibility === "visible" ? meta : null),
        )

      const writeMetaForCommand = (
        sessionId: string,
        change: (current: StoredSessionMeta) => StoredSessionMeta,
      ): Effect.Effect<ProtocolSessionMetadata, SessionCommandError> =>
        Effect.gen(function* () {
          const existing = yield* readMetaForCommand(sessionId)
          if (!existing) return yield* new SessionNotFound({ sessionId })
          const written = yield* storage.sessions.updateMeta(sessionId, (current) =>
            change(current ?? existing)).pipe(
            Effect.tapError((error) => Effect.logWarning("Session metadata write failed").pipe(
              Effect.annotateLogs({ sessionId, errorTag: error._tag }),
            )),
            Effect.mapError(() => new SessionMetadataWriteFailed({ sessionId })),
          )
          return yield* decodeSessionMetadata(written)
        })

      const setArchived = (sessionId: string, archived: boolean) =>
        writeMetaForCommand(sessionId, (current) => ({
          ...current,
          archived,
          ...(archived ? { pinnedAt: Option.none<string>() } : {}),
        }))

      return {
        createSession: Effect.fn("acn.session-lifecycle.create-session")(function* (
          cwd,
          sessionId,
          initial,
          options,
          draftOwnerId,
        ) {
          if (initial?._tag === "message" && !hasUserMessageContent(initial)) {
            return yield* new SessionStartFailed({
              sessionId: sessionId ?? "draft",
              reason: "Message content cannot be empty",
            })
          }
          if (initial?._tag === "goal" && !initial.objective.trim()) {
            return yield* new SessionStartFailed({
              sessionId: sessionId ?? "draft",
              reason: "Goal objective cannot be empty",
            })
          }

          // No initial: plain session creation (preload or explicit session id).
          // Return SessionMetadata directly wrapped as "created".
          if (!initial) {
            if (sessionId) {
              const existing = yield* readExisting(sessionId)
              if (existing) {
                return { _tag: "created" as const, metadata: existing }
              }
            }

            return yield* Effect.uninterruptibleMask((restore) => Effect.gen(function* () {
              const claim = yield* restore(drafts.claim({
                cwd: cwd ?? process.cwd(),
                sessionId,
                options,
                ownerId: draftOwnerId ?? null,
              }))
              const promoted = yield* Effect.exit(drafts.promote(claim))
              if (Exit.isFailure(promoted)) {
                yield* drafts.releaseClaim(claim)
                const failure = Cause.failureOption(promoted.cause)
                if (Option.isSome(failure)) return yield* failure.value
                return yield* Effect.failCause(promoted.cause)
              }
              return { _tag: "created" as const, metadata: promoted.value }
            }))
          }

          // With initial: check existence first (matching the !initial path),
          // then claim → sendUserMessage → promote.
          // Outcome-aware: distinguish message-sent-but-promote-failed from total failure.
          if (sessionId) {
            const live = yield* runtimeSnapshot(sessionId)
            if (live) {
              return yield* new SessionAlreadyExists({ sessionId })
            }
            const existing = yield* readExisting(sessionId)
            if (existing) {
              return yield* new SessionAlreadyExists({ sessionId })
            }
          }

          return yield* Effect.uninterruptibleMask((restore) => Effect.gen(function* () {
            const claim = yield* restore(drafts.claim({
              cwd: cwd ?? process.cwd(),
              sessionId,
              options,
              ownerId: draftOwnerId ?? null,
            }))

            const sendInitial = Effect.gen(function* () {
              if (initial?._tag === "message") {
                yield* commands.sendUserMessage({
                  sessionId: claim.sessionId,
                  messageId: Option.getOrUndefined(initial.messageId),
                  content: initial.content,
                  taskMode: initial.taskMode,
                  uploads: initial.uploads,
                  mentions: initial.mentions,
                })
              } else if (initial?._tag === "goal") {
                yield* commands.startGoal({ sessionId: claim.sessionId, objective: initial.objective })
              }
            })

            const sendResult = yield* Effect.exit(sendInitial)
            if (Exit.isFailure(sendResult)) {
              yield* drafts.releaseClaim(claim)
              const failure = Cause.failureOption(sendResult.cause)
              if (Option.isSome(failure)) {
                return { _tag: "failed" as const, error: sessionErrorMessage(failure.value) }
              }
              return yield* Effect.failCause(sendResult.cause)
            }

            // Once the initial command commits, promotion and rollback are an
            // atomic domain transition. Client interruption is observed only
            // after the draft reaches a terminal owned state.
            const promoteResult = yield* Effect.exit(drafts.promote(claim))
            if (Exit.isFailure(promoteResult)) {
              yield* drafts.releaseClaim(claim)
              const failure = Cause.failureOption(promoteResult.cause)
              if (Option.isNone(failure)) return yield* Effect.failCause(promoteResult.cause)
              return {
                _tag: "created_message_failed" as const,
                sessionId: claim.sessionId,
                error: sessionErrorMessage(failure.value),
              }
            }

            return { _tag: "created" as const, metadata: promoteResult.value }
          }))
        }),

        preloadSession: Effect.fn("acn.session-lifecycle.preload-session")(function* (
          cwd,
          options,
          draftOwnerId,
        ) {
          return yield* drafts.preload({ cwd, options, ownerId: draftOwnerId ?? null })
        }),

        releaseSessionPreload: Effect.fn("acn.session-lifecycle.release-session-preload")(
          function* (cwd, sessionId, options, draftOwnerId) {
            return yield* drafts.release({ cwd, sessionId, options, ownerId: draftOwnerId ?? null })
          },
        ),

        archiveSession: Effect.fn("acn.session-lifecycle.archive-session")(function* (sessionId) {
          return yield* setArchived(sessionId, true)
        }),

        restoreSession: Effect.fn("acn.session-lifecycle.restore-session")(function* (sessionId) {
          return yield* setArchived(sessionId, false)
        }),

        setSessionPinned: Effect.fn("acn.session-lifecycle.set-session-pinned")(function* (
          sessionId,
          pinned,
        ) {
          const pinnedAt = pinned
            ? Option.some(new Date(yield* Clock.currentTimeMillis).toISOString())
            : Option.none<string>()
          return yield* writeMetaForCommand(sessionId, (current) => ({
            ...current,
            archived: pinned ? false : current.archived,
            pinnedAt,
          }))
        }),

        deleteArchivedSession: Effect.fn("acn.session-lifecycle.delete-archived-session")(
          function* (sessionId) {
            const meta = yield* readMetaForCommand(sessionId)
            if (!meta) return yield* new SessionNotFound({ sessionId })
            if (!meta.archived) return yield* new SessionNotArchived({ sessionId })
            const removeDurableState = storage.sessions.deleteArchivedSession(sessionId).pipe(
              Effect.mapError((error) => new SessionOperationFailed({
                operation: `delete archived session ${sessionId}`,
                reason: error._tag,
              })),
              Effect.flatMap((deleted) => deleted
                ? Effect.void
                : Effect.fail(new SessionNotArchived({ sessionId }))),
            )
            // Runtime disposal composes in the runtime's own error domain; it is
            // translated exactly once at this command boundary.
            yield* runtime.deleteSession(sessionId, removeDurableState).pipe(
              Effect.catchAll((error): Effect.Effect<
                never,
                SessionNotFound | SessionNotArchived | SessionMetadataWriteFailed
              > =>
                error._tag === "SessionNotFound" || error._tag === "SessionNotArchived"
                  ? Effect.fail(error)
                  : Effect.logWarning("Archived session deletion failed").pipe(
                      Effect.annotateLogs({ sessionId, errorTag: error._tag }),
                      Effect.zipRight(Effect.fail(new SessionMetadataWriteFailed({ sessionId }))),
                    )),
            )
          },
        ),

        getSessionExecutionContext: Effect.fn("acn.session-lifecycle.get-session-execution-context")(
          function* (sessionId) {
            const live = yield* runtimeSnapshot(sessionId)
            if (live) {
              return { cwd: live.cwd, projectRoot: live.cwd, scratchpadPath: live.scratchpadPath }
            }
            const meta = yield* storage.sessions.readMeta(sessionId).pipe(
              Effect.mapError((error) => new SessionOperationFailed({
                operation: `read session metadata ${sessionId}`,
                reason: error._tag,
              })),
            )
            if (!meta) return yield* new SessionNotFound({ sessionId })
            return {
              cwd: meta.workingDirectory,
              projectRoot: meta.workingDirectory,
              scratchpadPath: storage.sessions.paths.sessionScratchpad(sessionId),
            }
          },
        ),

        getSessionCwd: Effect.fn("acn.session-lifecycle.get-session-cwd")(function* (sessionId) {
          const meta = yield* storage.sessions.readMeta(sessionId).pipe(
            Effect.mapError((error) => new SessionOperationFailed({
              operation: `read session metadata ${sessionId}`,
              reason: error._tag,
            })),
          )
          if (!meta) return yield* new SessionNotFound({ sessionId })
          return meta.workingDirectory
        }),
      } satisfies SessionLifecycleApi
    }),
  )
