import { Context, Effect, Layer } from "effect"
import type { SessionError } from "@magnitudedev/protocol"
import { AgentRuntime } from "./agent-runtime"
import { SessionStore } from "./session-store"

export interface SessionDestroyerApi {
  readonly destroySession: (
    sessionId: string,
    reason: string,
  ) => Effect.Effect<void, SessionError>
}

export class SessionDestroyer extends Context.Tag("SessionDestroyer")<
  SessionDestroyer,
  SessionDestroyerApi
>() {}

export const SessionDestroyerLive: Layer.Layer<
  SessionDestroyer,
  never,
  AgentRuntime | SessionStore
> =
  Layer.effect(
    SessionDestroyer,
    Effect.gen(function* () {
      const runtime = yield* AgentRuntime
      const store = yield* SessionStore

      return {
        destroySession: Effect.fn("acn.session-destroyer.destroy-session")(function* (sessionId, _reason) {
          yield* runtime.dispose(sessionId)
          yield* store.deleteSessionFiles(sessionId)
        }),
      }
    }),
  )
