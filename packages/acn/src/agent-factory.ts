import { Context, Effect, Layer, Scope } from "effect"
import type { CloseableScope } from "effect/Scope"
import {
  ChatPersistence,
  collectSessionContext,
  createCodingAgentSession,
  type CodingAgentSession,
} from "@magnitudedev/agent"
import { MagnitudeStorage, type StoredSessionMeta } from "@magnitudedev/storage"
import type { SessionError } from "@magnitudedev/protocol"
import { AcnChatPersistence } from "./agent-persistence"
import { toSessionError } from "./session-errors"
import type { SessionRuntimeOptions } from "./session-runtime-options"
import { ProviderClientRegistry, withModelSlotAdmission } from "./shared-client"
import { ModelSlotCoordinator } from "./model-slot-coordinator"

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
}): Layer.Layer<AgentFactory, never, MagnitudeStorage | ProviderClientRegistry | ModelSlotCoordinator> =>
  Layer.effect(
    AgentFactory,
    Effect.gen(function* () {
      const storage = yield* MagnitudeStorage
      const providerClients = yield* ProviderClientRegistry
      const modelSlots = yield* ModelSlotCoordinator

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
            const providerClient = withModelSlotAdmission(
              yield* providerClients.session(input.sessionId),
              modelSlots,
            )
            yield* Scope.addFinalizer(input.scope, providerClients.remove(input.sessionId))
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
            modelConfigurations: modelSlots.agentModelConfigurations,
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
