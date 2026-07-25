import { Clock, Context, Effect, Layer, Option } from 'effect'
import { ProviderClient, ProviderIdSchema, ProviderModelIdSchema, ReasoningEffortSchema, SlotIdSchema, type ProviderId, type ProviderModelId, type ReasoningEffort, type SlotId } from '@magnitudedev/sdk'
import { AmbientServiceTag, type AmbientService } from '@magnitudedev/event-core'
import type { RoleId } from '../agents/role-validation'
import { ConfigAmbient, getSlotConfig, getSlotConfigForRole, type SlotConfig } from '../ambient/config-ambient'
import { makeAgentBoundModel, type AgentBoundModel } from './agent-model'
import { TurnContextTag } from '../engine/turn-context'
import { ModelRequestActivityAmbient } from '../display'
import type { ModelRequestProgress } from '@magnitudedev/ai'
import type { PrepareModelRequest } from './model-request-preparation'

export type { AgentBoundModel } from './agent-model'

const MAX_TOOL_CALLS = 10

const LEADER_TRAITS = ['ATTENTIVE', 'STRATEGIC', 'PROACTIVE', 'RESPECTFUL', 'GROUNDED', 'INTROSPECTIVE', 'TASK'] as const

/**
 * Resolves models against the current event-engine configuration ambient.
 * The ambient is intentionally required at invocation time because it changes
 * during a live session and is supplied by the event engine, not this layer.
 *
 * @effect-expect-leaking AmbientService
 */
export interface AgentModelResolverService {
  readonly resolvePrimary: (roleId: RoleId, agentId?: string) => Effect.Effect<AgentBoundModel, never, AmbientService>
  readonly resolveSecondary: (agentId?: string) => Effect.Effect<AgentBoundModel, never, AmbientService>
  readonly resolveSlotConfig: (slot: SlotConfig, agentId?: string, roleId?: RoleId) => Effect.Effect<AgentBoundModel, never, AmbientService>
}

/** @effect-expect-leaking AmbientService */
export class AgentModelResolver extends Context.Tag('AgentModelResolver')<
  AgentModelResolver,
  AgentModelResolverService
>() {}

export interface ReasoningEffortFallbackInput {
  readonly slotId: SlotId
  readonly providerId: ProviderId
  readonly providerModelId: ProviderModelId
  readonly requested: ReasoningEffort
  readonly fallback: ReasoningEffort
}

export const AgentModelResolverLive = (
  debug?: boolean,
  applyReasoningEffortFallback: (input: ReasoningEffortFallbackInput) => Effect.Effect<void, unknown> = () => Effect.void,
  prepareModelRequest: PrepareModelRequest = () => Effect.void,
) =>
  Layer.effect(
    AgentModelResolver,
    Effect.gen(function* () {
      const client = yield* ProviderClient

      const { preferProvider, disableTraits } = client.runtimeConfig

      function resolveFromSlot(
        slotConfig: SlotConfig,
        agentId: string,
        options: {
          readonly roleId?: RoleId | null
          readonly traits?: readonly string[]
          readonly maxToolCalls?: number
          readonly maxTokensOverride?: number
        },
      ): Effect.Effect<AgentBoundModel, never, AmbientService> {
        return Effect.gen(function* () {
          const ambient = yield* AmbientServiceTag
          const defaults = {
            maxTokens: options.maxTokensOverride ?? slotConfig.profile.maxOutputTokens,
            reasoningEffort: ReasoningEffortSchema.make(slotConfig.reasoningEffort),
          }

          const providerId = ProviderIdSchema.make(slotConfig.providerId)
          const providerModelId = ProviderModelIdSchema.make(slotConfig.providerModelId)
          const slotId = SlotIdSchema.make(slotConfig.slotId)
          const requestAttribution = client.requestAttribution(
            providerId,
            providerModelId,
            slotId,
          )
          const reportProgress = (progress: ModelRequestProgress) =>
            Effect.gen(function* () {
              const turn = yield* Effect.serviceOption(TurnContextTag)
              if (Option.isSome(turn)) {
                const observedAt = yield* Clock.currentTimeMillis
                yield* ambient.update(ModelRequestActivityAmbient, {
                  turn: turn.value,
                  progress,
                  observedAt,
                })
              }
            })
          const rawModel = yield* client.resolveModel(providerId, providerModelId, {
            defaults,
            requestAttribution: {
              ...requestAttribution,
              requestProgress: reportProgress,
            },
            reasoningEffortFallback: (requested, fallback) => applyReasoningEffortFallback({
              slotId,
              providerId,
              providerModelId,
              requested,
              fallback,
            }),
            agentId,
            ...(options.roleId ? { roleId: options.roleId } : {}),
            ...(options.traits ? { traits: options.traits } : {}),
            ...(preferProvider ? { preferProvider } : {}),
          })

          return makeAgentBoundModel({
            rawModel,
            prepareRequest: prepareModelRequest({
              slotId,
              providerId,
              providerModelId,
              reportProgress,
            }),
            clearRequestProgress: reportProgress({
              phase: 'cleared',
              requestId: null,
            }),
            modelSource: { slotId: slotConfig.slotId },
            modelId: slotConfig.providerModelId,
            providerId: slotConfig.providerId,
            profile: slotConfig.profile,
            debug: debug ?? false,
            agentId,
            roleId: options.roleId ?? null,
            ...(options.maxToolCalls !== undefined ? { maxToolCalls: options.maxToolCalls } : {}),
          })
        })
      }

      return {
        resolveSlotConfig: (slot, agentId, roleId) => resolveFromSlot(slot, agentId ?? '000000000000', {
          roleId,
          maxToolCalls: MAX_TOOL_CALLS,
          traits: roleId === 'leader' && !disableTraits ? LEADER_TRAITS : undefined,
        }),
        resolvePrimary: (roleId: RoleId, agentId?: string) =>
          Effect.gen(function* () {
            const ambientService = yield* AmbientServiceTag
            const configState = ambientService.getValue(ConfigAmbient)
            const slotConfig = getSlotConfigForRole(configState, roleId)
            return yield* resolveFromSlot(slotConfig, agentId ?? '000000000000', {
              roleId,
              maxToolCalls: MAX_TOOL_CALLS,
              traits: roleId === 'leader' && !disableTraits ? LEADER_TRAITS : undefined,
            })
          }),

        resolveSecondary: (agentId?: string) =>
          Effect.gen(function* () {
            const ambientService = yield* AmbientServiceTag
            const configState = ambientService.getValue(ConfigAmbient)
            const slotConfig = getSlotConfig(configState, 'secondary')
            return yield* resolveFromSlot(slotConfig, agentId ?? 'secondary', {
              maxToolCalls: MAX_TOOL_CALLS,
            })
          }),
      }
    }),
  )
