import { Context, Effect, Layer, Option } from "effect"
import { resolve } from "path"
import {
  SessionAlreadyExists,
  SessionNotFound,
  SessionStartFailed,
  type CreateSessionInitial,
  type CreateSessionResult,
  type ListSessionsResult,
  type SessionError,
  type SessionCwdSummary,
  type SessionMetadata as ProtocolSessionMetadata,
  type SessionOptions,
} from "@magnitudedev/protocol"
import { AgentRuntime } from "./agent-runtime"
import { SessionDrafts } from "./session-drafts"
import { SessionCommands } from "./session-commands"
import { sessionErrorMessage } from "./session-errors"
import { SessionStore } from "./session-store"
import type { SessionExecutionContext } from "./session-types"
import type { ResidentSessionSnapshot } from "./agent-runtime"

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
    options?: SessionOptions,
    draftOwnerId?: string | null,
  ) => Effect.Effect<void, SessionError>
  readonly listSessions: (
    options?: { readonly cwd?: string; readonly query?: string; readonly cursor?: string; readonly limit?: number }
  ) => Effect.Effect<ListSessionsResult, SessionError>
  readonly listSessionCwds: () => Effect.Effect<ReadonlyArray<SessionCwdSummary>, SessionError>
  readonly getSessionInfo: (sessionId: string) => Effect.Effect<ProtocolSessionMetadata, SessionError>
  readonly deleteSession: (sessionId: string) => Effect.Effect<void, SessionError>
  readonly getSessionExecutionContext: (sessionId: string) => Effect.Effect<SessionExecutionContext, SessionError>
  readonly getSessionCwd: (sessionId: string) => Effect.Effect<string, SessionError>
}

export class SessionLifecycle extends Context.Tag("SessionLifecycle")<
  SessionLifecycle,
  SessionLifecycleApi
>() {}

const toMetadata = (entry: ResidentSessionSnapshot, stored: ProtocolSessionMetadata | null): ProtocolSessionMetadata => ({
  sessionId: entry.sessionId,
  title: entry.title,
  cwd: entry.cwd,
  createdAt: entry.createdAt,
  updatedAt: stored?.updatedAt ?? entry.updatedAt,
  messageCount: stored?.messageCount ?? 0,
  lastMessage: stored?.lastMessage ?? null,
})

export const SessionLifecycleLive: Layer.Layer<
  SessionLifecycle,
  never,
  AgentRuntime | SessionCommands | SessionDrafts | SessionStore
> =
  Layer.effect(
    SessionLifecycle,
    Effect.gen(function* () {
    const runtime = yield* AgentRuntime
      const commands = yield* SessionCommands
      const drafts = yield* SessionDrafts
    const store = yield* SessionStore

    const residentSnapshot = (sessionId: string) =>
      runtime.residentSessions.pipe(
        Effect.map((sessions) => sessions.find((session) => session.sessionId === sessionId)),
      )
      return {
        createSession: Effect.fn("acn.session-lifecycle.create-session")(function* (cwd, sessionId, initial, options, draftOwnerId) {
          if (initial?._tag === "message" && !initial.content.trim()) {
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
              const live = yield* residentSnapshot(sessionId)
              if (live) {
                const stored = yield* store.readProtocolMeta(sessionId)
                if (!stored) return yield* new SessionNotFound({ sessionId })
                return { _tag: "created" as const, metadata: toMetadata(live, stored) }
              }
              const existing = yield* store.readProtocolMeta(sessionId)
              if (existing) {
                return { _tag: "created" as const, metadata: existing }
              }
            }

            const claim = yield* drafts.claim({
              cwd: cwd ? resolve(cwd) : process.cwd(),
              sessionId,
              options,
              ownerId: draftOwnerId ?? null,
            })
            const promoted = yield* drafts.promote(claim).pipe(
              Effect.catchAll((error) =>
                drafts.releaseClaim(claim).pipe(
                  Effect.andThen(Effect.fail(error)),
                ),
              ),
            )
            return { _tag: "created" as const, metadata: promoted }
          }

          // With initial: check existence first (matching the !initial path),
          // then claim → sendUserMessage → promote.
          // Outcome-aware: distinguish message-sent-but-promote-failed from total failure.
          if (sessionId) {
            const live = yield* residentSnapshot(sessionId)
            if (live) {
              return yield* new SessionAlreadyExists({ sessionId })
            }
            const existing = yield* store.readProtocolMeta(sessionId)
            if (existing) {
              return yield* new SessionAlreadyExists({ sessionId })
            }
          }

          const claim = yield* drafts.claim({
            cwd: cwd ? resolve(cwd) : process.cwd(),
            sessionId,
            options,
            ownerId: draftOwnerId ?? null,
          })

          const sendInitial = Effect.gen(function* () {
            if (initial?._tag === "message") {
              yield* commands.sendUserMessage({
                sessionId: claim.sessionId,
                messageId: Option.getOrUndefined(initial.messageId),
                content: initial.content,
                taskMode: initial.taskMode,
                imageAttachments: initial.imageAttachments,
                mentions: initial.mentions,
              })
            } else if (initial?._tag === "goal") {
              yield* commands.startGoal({ sessionId: claim.sessionId, objective: initial.objective })
            }
          })

          // sendUserMessage failure: message was NOT sent. Release claim
          // (reverts to ready, scope stays open) and return failed outcome.
          // The empty draft will be swept.
          const sendResult = yield* Effect.either(sendInitial)
          if (sendResult._tag === "Left") {
            yield* drafts.releaseClaim(claim)
            return { _tag: "failed" as const, error: sessionErrorMessage(sendResult.left) }
          }

          // Message was sent. Now promote. If promote fails, the message IS
          // in the agent — return created_message_failed so the client keeps
          // the optimistic message and selects the session.
          const promoteResult = yield* Effect.either(drafts.promote(claim))
          if (promoteResult._tag === "Left") {
            yield* drafts.releaseClaim(claim)
            return {
              _tag: "created_message_failed" as const,
              sessionId: claim.sessionId,
              error: sessionErrorMessage(promoteResult.left),
            }
          }

          return { _tag: "created" as const, metadata: promoteResult.right }
        }),
        preloadSession: Effect.fn("acn.session-lifecycle.preload-session")(function* (cwd, options, draftOwnerId) {
          return yield* drafts.preload({
            cwd,
            options,
            ownerId: draftOwnerId ?? null,
          })
        }),
        releaseSessionPreload: Effect.fn("acn.session-lifecycle.release-session-preload")(function* (cwd, options, draftOwnerId) {
          return yield* drafts.release({
            cwd,
            options,
            ownerId: draftOwnerId ?? null,
          })
        }),
        listSessions: (options) => store.listProtocolMetas(options),
        listSessionCwds: () => store.listSessionCwds(),
        getSessionInfo: Effect.fn("acn.session-lifecycle.get-session-info")(function* (sessionId) {
          const live = yield* residentSnapshot(sessionId)
          if (live) {
            const stored = yield* store.readProtocolMeta(sessionId)
            if (!stored) return yield* new SessionNotFound({ sessionId })
            return toMetadata(live, stored)
          }
          const meta = yield* store.readProtocolMeta(sessionId)
          if (!meta) return yield* new SessionNotFound({ sessionId })
          return meta
        }),
        deleteSession: Effect.fn("acn.session-lifecycle.delete-session")(function* (sessionId) {
          const live = yield* residentSnapshot(sessionId)
          const meta = yield* store.readMeta(sessionId)
          if (!live && !meta) return yield* new SessionNotFound({ sessionId })
          yield* runtime.deleteSession(sessionId, store.deleteSessionFiles(sessionId))
        }),
        getSessionExecutionContext: Effect.fn("acn.session-lifecycle.get-session-execution-context")(function* (sessionId) {
          const live = yield* residentSnapshot(sessionId)
          if (live) {
            return { cwd: live.cwd, projectRoot: live.cwd, scratchpadPath: live.scratchpadPath }
          }
          return yield* store.getExecutionContext(sessionId)
        }),
        getSessionCwd: Effect.fn("acn.session-lifecycle.get-session-cwd")(function* (sessionId) {
          return (yield* store.getExecutionContext(sessionId)).cwd
        }),
      }
    }),
  )
