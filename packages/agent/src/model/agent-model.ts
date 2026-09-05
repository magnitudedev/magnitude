import { Context, Effect, Option, Stream } from 'effect'
import {
  TraceListener,
  type BoundModel,
  type BaseCallOptions,
  type ToolCallId,
  type ToolDefinition,
  type StreamStartFailure,
  type ModelStreamEvent,
} from '@magnitudedev/ai'
import type { IcnModelPreparation } from '@magnitudedev/providers/client'
import type { ModelRequestActivity } from './model-request-activity'
import type { SlotId } from '@magnitudedev/roles'
import { buildMaxToolCallsGrammar } from './tool-call-grammar'
import { Fork } from '@magnitudedev/event-core'
import {
  getTraceSessionId,
  writeTrace,
  type AgentCallType,
  type AgentTraceActor,
  type AgentTraceScope,
} from '@magnitudedev/tracing'
import { TurnContextTag } from '../engine/turn-context'
import type { RoleId } from '../agents/role-validation'
import { createId } from '../util/id'
import type { AgentModelStartFailure } from './model-request-preparation'

const { ForkContext } = Fork

export type AgentModelOperationKind =
  | 'compact'
  | 'observer'
  | 'advisor'
  | 'autopilot'
  | 'image'
  | 'background'

export interface AgentModelOperationContext {
  readonly operationKind: AgentModelOperationKind
  readonly operationId?: string
  readonly relatedTurnId?: string
  readonly relatedMessageId?: string
  readonly parentOperationId?: string
  readonly chainId?: string
  readonly forkId?: string | null
}

export const AgentModelOperationContextTag = Context.GenericTag<AgentModelOperationContext>(
  'AgentModelOperationContext',
)

export type ModelSource = { readonly slotId: SlotId }

/**
 * A model bound and ready for inference. Provider-agnostic — the provider
 * baked in its specific options at bind time. The agent layer adds tracing
 * and grammar wrapping on top, operating on universal `BaseCallOptions`.
 */
export interface AgentBoundModel {
  readonly model: BoundModel<BaseCallOptions, AgentModelStartFailure>
  readonly modelSource: ModelSource
  readonly modelId: string
  readonly modelDisplayName: string
  /**
   * Provider ID, carried for event attribution/logging only.
   *
   * DO NOT use this for branching, switch statements, or conditional logic.
   * Any logic that would switch on provider ID represents a failure to
   * capture a common abstraction in the provider contract. If you need
   * provider-specific behavior, extend the Provider interface — do not
   * check this field.
   */
  readonly providerId: string
  readonly profile: { readonly contextWindow: number; readonly maxOutputTokens: number }
  readonly maxToolCalls?: number
}

export interface AgentBoundModelConfig {
  readonly rawModel: BoundModel<BaseCallOptions, StreamStartFailure, IcnModelPreparation>
  readonly reportActivity?: (activity: ModelRequestActivity) => Effect.Effect<void>
  readonly modelId: string
  readonly modelDisplayName: string
  readonly modelSource: ModelSource
  readonly providerId: string
  readonly profile: { readonly contextWindow: number; readonly maxOutputTokens: number }
  readonly debug: boolean
  readonly agentId: string
  readonly maxToolCalls?: number
  readonly roleId?: RoleId | null
}

type PreparationUpdate = Extract<
  ModelStreamEvent<IcnModelPreparation>,
  { readonly _tag: 'preparation_update' }
>

const isPreparationUpdate = (
  event: ModelStreamEvent<IcnModelPreparation>,
): event is PreparationUpdate => event._tag === 'preparation_update'

/**
 * When `maxToolCalls` is configured and tools are present, override `toolChoice`
 * with a bounded-countdown grammar that limits tool calls per response.
 */
type ToolChoiceMode = "auto" | "required"

type CallOptionsWithToolIds = BaseCallOptions & {
  readonly generateToolCallId?: () => ToolCallId
}

/**
 * Determine the tool-choice mode from a tool choice value.
 * Returns None for "none", grammar, named-function, allowed-tools —
 * those preserve caller intent and don't need grammar wrapping.
 */
function toolChoiceMode(toolChoice: Option.Option<BaseCallOptions["toolChoice"]>): Option.Option<ToolChoiceMode> {
  return Option.gen(function* () {
    const tc = yield* toolChoice
    if (tc === "auto") return "auto"
    if (tc === "required") return "required"
    return yield* Option.none()
  })
}

interface GrammarToolChoice {
  readonly type: "grammar"
  readonly grammar: string
}

function computeMaxToolCallsGrammar(
  tools: readonly ToolDefinition[],
  toolChoice: Option.Option<BaseCallOptions["toolChoice"]>,
  maxToolCalls: Option.Option<number>,
): Option.Option<GrammarToolChoice> {
  return Option.gen(function* () {
    const max = yield* maxToolCalls
    if (max < 1 || tools.length === 0) return yield* Option.none()
    const mode = yield* toolChoiceMode(toolChoice)
    const toolNames = tools.map((t) => t.name)
    const grammar = buildMaxToolCallsGrammar(toolNames, max, mode)
    if (grammar === "") return yield* Option.none()
    return { type: "grammar", grammar }
  })
}

function defaultCallType(
  operation: AgentModelOperationContext | null,
): AgentCallType {
  if (operation?.operationKind === 'compact') return 'compact'
  if (operation?.operationKind === 'image') return 'image'
  if (operation?.operationKind === 'observer') return 'observer'
  if (operation?.operationKind === 'autopilot') return 'autopilot'
  if (operation?.operationKind === 'advisor') return 'advisor'
  return 'chat'
}

function defaultOperationKind(callType: AgentCallType): AgentModelOperationKind {
  switch (callType) {
    case 'compact':
      return 'compact'
    case 'advisor':
      return 'advisor'
    case 'observer':
      return 'observer'
    case 'autopilot':
      return 'autopilot'
    case 'image':
      return 'image'
    case 'chat':
    case 'extract-memory-diff':
      return 'background'
  }
}

function deriveActor(input: {
  readonly config: Pick<AgentBoundModelConfig, 'agentId' | 'roleId'>
  readonly turnForkId: string | null | undefined
  readonly fork: Fork.ForkContextService | null
  readonly operation: AgentModelOperationContext | null
}): AgentTraceActor {
  const forkId = input.turnForkId
    ?? input.fork?.forkId
    ?? input.operation?.forkId
    ?? null

  return {
    agentId: input.config.agentId,
    forkId,
    roleId: input.config.roleId ?? null,
  }
}

function deriveScope(input: {
  readonly callType: AgentCallType
  readonly turn: { readonly turnId: string; readonly chainId: string; readonly forkId: string | null } | null
  readonly operation: AgentModelOperationContext | null
}): AgentTraceScope {
  if (input.turn) {
    return {
      kind: 'turn',
      turnId: input.turn.turnId,
      chainId: input.turn.chainId,
    }
  }

  const operationKind = input.operation?.operationKind ?? defaultOperationKind(input.callType)
  return {
    kind: 'operation',
    operationId: input.operation?.operationId ?? createId(),
    operationKind,
    ...(input.operation?.chainId !== undefined ? { chainId: input.operation.chainId } : {}),
    ...(input.operation?.relatedTurnId !== undefined ? { relatedTurnId: input.operation.relatedTurnId } : {}),
    ...(input.operation?.relatedMessageId !== undefined ? { relatedMessageId: input.operation.relatedMessageId } : {}),
    ...(input.operation?.parentOperationId !== undefined ? { parentOperationId: input.operation.parentOperationId } : {}),
    ...(input.operation?.forkId !== undefined ? { forkId: input.operation.forkId } : {}),
  }
}

// =============================================================================
// Agent Bound Model — single factory, provider-agnostic
// =============================================================================

export function makeAgentBoundModel(config: AgentBoundModelConfig): AgentBoundModel {
  const model: BoundModel<BaseCallOptions, AgentModelStartFailure> = {
    stream: (prompt, tools, options) =>
      Effect.gen(function* () {
        const callOptions = options as CallOptionsWithToolIds | undefined
        const grammarChoice = computeMaxToolCallsGrammar(
          tools,
          callOptions?.toolChoice !== undefined ? Option.some(callOptions.toolChoice) : Option.none(),
          config.maxToolCalls !== undefined ? Option.some(config.maxToolCalls) : Option.none(),
        )
        const effectiveOptions = Option.match(grammarChoice, {
          onSome: (gc) => ({ ...callOptions, toolChoice: gc }),
          onNone: () => callOptions,
        })
        const streamEffect = Effect.gen(function* () {
          let requestId: string | null = null
          yield* (config.reportActivity?.({ _tag: 'Starting', requestId }) ?? Effect.void)
          let accepted = false
          const streamResult = yield* config.rawModel.stream(prompt, tools, effectiveOptions).pipe(
            Effect.tap((result) => Effect.sync(() => {
              accepted = true
              requestId = result.requestId
            })),
            Effect.ensuring(Effect.suspend(() =>
              accepted
                ? Effect.void
                : config.reportActivity?.({ _tag: 'Ended', requestId }) ?? Effect.void)),
          )
          let reportedStreaming = false
          const observedEvents = streamResult.events.pipe(
            Stream.tap((event) => {
              if (isPreparationUpdate(event)) {
                if (event.requestId !== null) requestId = event.requestId
                return config.reportActivity?.({
                  _tag: 'Preparing',
                  preparation: event.preparation,
                  requestId,
                }) ?? Effect.void
              }

              if (event._tag === 'stream_end' || reportedStreaming) return Effect.void
              reportedStreaming = true
              return config.reportActivity?.({ _tag: 'Streaming', requestId }) ?? Effect.void
            }),
          )
          const events = Stream.unwrapScoped(observedEvents.pipe(
            Stream.partition(isPreparationUpdate),
            Effect.map(([responses, preparation]) =>
              Stream.merge(responses, preparation.pipe(Stream.drain))),
          )).pipe(
            Stream.ensuring(Effect.suspend(() =>
              config.reportActivity?.({ _tag: 'Ended', requestId }) ?? Effect.void)),
          )
          return { ...streamResult, events }
        })

        if (!config.debug) return yield* streamEffect

        const sessionId = getTraceSessionId()
        if (sessionId === null) return yield* streamEffect

        const turnOption = yield* Effect.serviceOption(TurnContextTag)
        const forkOption = yield* Effect.serviceOption(ForkContext)
        const operationOption = yield* Effect.serviceOption(AgentModelOperationContextTag)

        const turn = turnOption._tag === 'Some' ? turnOption.value : null
        const fork = forkOption._tag === 'Some' ? forkOption.value : null
        const operation = operationOption._tag === 'Some' ? operationOption.value : null
        const callType = defaultCallType(operation)
        const actor = deriveActor({
          config,
          turnForkId: turn?.forkId,
          fork,
          operation,
        })
        const scope = deriveScope({ callType, turn, operation })

        return yield* streamEffect.pipe(
          Effect.provideService(TraceListener, {
            onTrace: (trace) => {
              writeTrace({
                ...trace,
                traceId: createId(),
                sessionId,
                actor,
                callType,
                scope,
              })
            },
          }),
        )
      }),
  }

  return {
    model,
    modelSource: config.modelSource,
    modelId: config.modelId,
    modelDisplayName: config.modelDisplayName,
    providerId: config.providerId,
    profile: config.profile,
    ...(config.maxToolCalls !== undefined ? { maxToolCalls: config.maxToolCalls } : {}),
  }
}
