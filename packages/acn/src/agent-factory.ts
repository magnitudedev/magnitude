import { Context, Effect, Layer, Scope } from "effect"
import type { CloseableScope } from "effect/Scope"
import {
  ChatPersistence,
  collectSessionContext,
  createCodingAgentSession,
  type CodingAgentSession,
} from "@magnitudedev/agent"
import { createProviderClient, type ProviderClientShape } from "@magnitudedev/sdk"
import { MagnitudeStorage, type StoredSessionMeta } from "@magnitudedev/storage"
import type { SessionError } from "@magnitudedev/protocol"
import { AcnChatPersistence } from "./agent-persistence"
import { toSessionError } from "./session-errors"
import type { SessionRuntimeOptions } from "./session-runtime-options"

export interface AgentFactoryApi {
  readonly createSession: (input: {
    readonly sessionId: string
    readonly cwd: string
    readonly scope: CloseableScope
    readonly options: SessionRuntimeOptions
    readonly visibility?: StoredSessionMeta["visibility"]
  }) => Effect.Effect<CodingAgentSession, SessionError>
}

export class AgentFactory extends Context.Tag("AgentFactory")<
  AgentFactory,
  AgentFactoryApi
>() {}

export const AgentFactoryLive = (options: {
  readonly debug: boolean
  readonly version: string
}): Layer.Layer<AgentFactory, never, MagnitudeStorage> =>
  Layer.effect(
    AgentFactory,
    Effect.gen(function* () {
      const storage = yield* MagnitudeStorage

      return {
        createSession: Effect.fn("acn.agent-factory.create-session")(function* (input) {
          const prepared = yield* Effect.gen(function* () {
            const persistence = new AcnChatPersistence(
              storage,
              input.cwd,
              input.sessionId,
              options.version,
              input.visibility ?? "visible",
            )
            const persistenceLayer = Layer.succeed(ChatPersistence, persistence)
            const sessionContext = yield* Effect.tryPromise(() => collectSessionContext({
              cwd: input.cwd,
              storage,
            })).pipe(
              Effect.mapError((cause) => toSessionError(input.sessionId, cause)),
            )
            const apiKey = yield* storage.auth.get("magnitude").pipe(
              Effect.map((auth) => auth?.type === "api" ? auth.key : null),
            )
            const magnitudeApiKey = apiKey
              || process.env.MAGNITUDE_API_KEY
              || process.env.MAGNITUDE_LOCAL_API_KEY
            if (!magnitudeApiKey) return yield* toSessionError(input.sessionId, new Error("No Magnitude API key found"))
            const providerClient: ProviderClientShape = createProviderClient({
              apiKey: magnitudeApiKey,
              sessionId: input.sessionId,
            })
            return { persistenceLayer, sessionContext, providerClient }
          }).pipe(
            Effect.mapError((cause) => toSessionError(input.sessionId, cause)),
          )

          return yield* createCodingAgentSession({
            persistence: prepared.persistenceLayer,
            storage,
            debug: options.debug,
            sessionContext: prepared.sessionContext,
            sessionId: input.sessionId,
            providerClient: prepared.providerClient,
            disableShellSafeguards: input.options.disableShellSafeguards,
            disableCwdSafeguards: input.options.disableCwdSafeguards,
            atifPath: input.options.atifPath ?? undefined,
            solo: input.options.solo,
            systemPromptOverride: input.options.systemPromptOverride ?? undefined,
            headless: input.options.headless,
          }).pipe(
            Effect.provideService(Scope.Scope, input.scope),
            Effect.mapError((cause) => toSessionError(input.sessionId, cause)),
          )
        }),
      }
    }),
  )
