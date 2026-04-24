/**
 * Cortex Worker (Forked)
 *
 * Handles LLM streaming and xml-act execution coordination.
 * Uses Worker.defineForked for per-fork concurrent execution:
 * - Each fork gets its own fiber (parallel LLM calls)
 * - Fork completion automatically interrupts that fork's fiber
 * - Root fork (null) always runs
 *
 * Responds to turn_started events by:
 * 1. Getting memory context for the fork
 * 2. Building system prompt (XML_ACT_PROTOCOL + role + tool docs)
 * 3. Streaming LLM response
 * 4. Executing via xml-act runtime (execution manager)
 * 5. Publishing turn_completed
 */

import { Effect, Stream, Either } from 'effect'
import { Worker, AmbientServiceTag } from '@magnitudedev/event-core'

import { LEAD_YIELD_STOP_SEQUENCES, SUBAGENT_YIELD_STOP_SEQUENCES, LEAD_YIELD_TAGS, SUBAGENT_YIELD_TAGS, type GrammarBuildOptions, type TurnEngineCrash } from '@magnitudedev/xml-act'
import { logger } from '@magnitudedev/logger'

import { ContextLimitExceeded, AuthFailed, TransportError as ProviderTransportError, ParseError as ProviderParseError, SubscriptionRequired, UsageLimitExceeded } from '@magnitudedev/providers'
import type { ModelError } from '@magnitudedev/providers'
import { drainTurnEventStream } from './turn-event-drain'
import type { ChatMessage } from '@magnitudedev/llm-core'
import { BamlClientHttpError, BamlValidationError } from '@magnitudedev/llm-core'
import { Image as BamlImage } from '@boundaryml/baml'
import type { ObservationPart } from '@magnitudedev/roles'
import { buildAckTurns } from '../prompts/protocol'
import { renderSystemPrompt } from '../prompts/system-prompt'
import { ContentPart } from '../content'
import type { AppEvent, TurnOutcome } from '../events'

import { createTurnStream } from '../execution/turn-stream'
import { TurnError as TurnErrorCtor } from '../execution/types'
import type { TurnError } from '../execution/types'
import { MemoryProjection, getView } from '../projections/memory'

import { LLMMessage } from '../projections/memory'
import { CompactionProjection } from '../projections/compaction'
import { SessionContextProjection } from '../projections/session-context'
import { AgentStatusProjection, getAgentByForkId } from '../projections/agent-status'

import { TurnProjection } from '../projections/turn'
import { ExecutionManager } from '../execution/types'
import { getAgentDefinition, getForkInfo } from '../agents/registry'
import { generateToolGrammar } from '../tools/tool-registry'
import { buildResolvedToolSet } from '../tools/resolved-toolset'

import { ModelResolver, CodingAgentChat } from '@magnitudedev/providers'
import { withTraceScope } from '../tracing/scoped-tracer'
import { buildInterruptedTurnCompleted } from '../util/interrupt-utils'
import { ConfigAmbient, getSlotConfig } from '../ambient/config-ambient'
import { SkillsAmbient } from '../ambient/skills-ambient'
import {
  authReconnectMessage,
  buildGeneralErrorPayload,
  classifyRetryability,
  isAuthReconnectMessage,
  resolveFailureMessage,
} from './cortex-auth'

type TaggedCause = {
  readonly _tag: string
  readonly message?: string
}

function isTaggedCause(cause: unknown): cause is TaggedCause {
  return typeof cause === 'object' && cause !== null && '_tag' in cause
}

function toLLMContent(parts: ContentPart[]): (BamlImage | string)[] {
  return parts.map(part => {
    switch (part.type) {
      case 'text': return part.text
      case 'image': return BamlImage.fromBase64(part.mediaType, part.base64)
    }
  })
}

function toBamlMessages(messages: LLMMessage[]): ChatMessage[] {
  return messages.map(m => ({
    role: m.role,
    content: toLLMContent(m.content)
  }))
}

// =============================================================================
// Worker
// =============================================================================

export const Cortex = Worker.defineForked<AppEvent>()({
  name: 'Cortex',

  forkLifecycle: {
    activateOn: 'agent_created',
    completeOn: ['agent_killed', 'subagent_user_killed', 'subagent_idle_closed'],
  },

  eventHandlers: {
    subagent_user_killed: (event) => Effect.gen(function* () {
      if (event.forkId === null) return
      return yield* Effect.interrupt
    }),
    subagent_idle_closed: (event) => Effect.gen(function* () {
      if (event.forkId === null) return
      return yield* Effect.interrupt
    }),
    turn_started: (event, publish, read) => {
      const { forkId, turnId, chainId } = event

      let resolvedProviderId: string | null = null
      let resolvedModelId: string | null = null
      return Effect.gen(function* () {
        const sessionCtx = yield* read(SessionContextProjection)
        const agentState = yield* read(AgentStatusProjection)
        const turnState = yield* read(TurnProjection, forkId)
        const triggeredByUser =
          (turnState._tag === 'active' || turnState._tag === 'interrupting')
            ? turnState.triggeredByUser
            : false
        const forkInfo = getForkInfo(agentState, forkId)
        if (!forkInfo) return
        const { variant, slot: modelSlot } = forkInfo
        const agentDef = getAgentDefinition(variant)
        const agentInstance = forkId ? getAgentByForkId(agentState, forkId) : null
        const timezone = sessionCtx.context?.timezone ?? null

        const runtime = yield* ModelResolver



        // Run agent observables
        const execManager = yield* ExecutionManager
        const observations: ObservationPart[] = []
        const boundObs = execManager.getObservables(forkId)
        for (const obs of boundObs) {
          const parts = yield* obs.observe()
          observations.push(...parts)
        }

        // Publish observations so memory projection includes them
        if (observations.length > 0) {
          yield* publish({
            type: 'observations_captured',
            forkId,
            turnId,
            parts: observations,
          })
        }

        // 2. Build system prompt with runtime protocol/tool-doc substitution
        const ambientService = yield* AmbientServiceTag
        const skills = ambientService.getValue(SkillsAmbient)

        // Build ResolvedToolSet for this slot — single decision site for tool availability
        const configState = ambientService.getValue(ConfigAmbient)
        const toolSet = buildResolvedToolSet(agentDef, configState, modelSlot)

        // Build messages array (now includes observations in system inbox)
        const forkMemory = yield* read(MemoryProjection)
        const chatMessages: ChatMessage[] = toBamlMessages(getView(forkMemory.messages, timezone, 'agent'))
        const systemPrompt = renderSystemPrompt(agentDef, skills, toolSet)

        logger.info({ variant, forkId, turnId }, '[Cortex] Executing turn via xml-act')

        // Step 1: Ensure model ready
        const resolveResult = yield* runtime.resolve(modelSlot).pipe(Effect.either)
        if (Either.isLeft(resolveResult)) {
          const e = resolveResult.left
          const message = resolveFailureMessage(e)
          yield* publish({ type: 'turn_unexpected_error', forkId, turnId, message })
          return
        }
        const boundModel = resolveResult.right
        resolvedProviderId = boundModel.model.providerId
        resolvedModelId = boundModel.model.id

        // 2b. Provide input token estimate so provider can clamp max_tokens
        const compactionState = yield* read(CompactionProjection, forkId)

        // 3. Build and consume the turn event stream
        // Check if grammar is disabled via env var (MAGNITUDE_ENABLE_GRAMMAR)
        const grammarEnabled = ((): boolean => {
          const envValue = process.env.MAGNITUDE_ENABLE_GRAMMAR
          if (envValue === undefined) return true // Default: enabled
          const normalized = envValue.toLowerCase().trim()
          return normalized !== '0' && normalized !== 'false' && normalized !== ''
        })()
        const grammarSafe = boundModel.model.supportsGrammar !== false
        // Select stop sequences and yield tags based on protocol role
        const isSubagent = agentDef.protocolRole === 'subagent'
        // Stop sequences not passed for now — yield tags are self-closing and would be
        // cut from the stream if used as stop sequences, losing the decision info.
        // Grammar-constrained models end naturally after yield. Non-grammar models
        // are handled by PostYieldObserver runaway detection.
        // const stopSequences = isSubagent ? [...SUBAGENT_YIELD_STOP_SEQUENCES] : [...LEAD_YIELD_STOP_SEQUENCES]
        const yieldTags = isSubagent ? [...SUBAGENT_YIELD_TAGS] : [...LEAD_YIELD_TAGS]

        const grammarOptions: GrammarBuildOptions = {
          // Temporarily disabled: forced message to user on user-triggered turns
          // causes issues with XML format parsing
          // ...(triggeredByUser ? { requiredMessageTo: 'user', maxLenses: agentDef.lenses.length } : {}),
          yieldTags,
        }

        const toolGrammar = grammarEnabled && grammarSafe
          ? generateToolGrammar(toolSet, grammarOptions)
          : undefined
        const turnStream = createTurnStream((sink) => Effect.gen(function* () {
          const ackTurns = buildAckTurns(agentDef.lenses, agentDef.defaultRecipient)
          const cs = yield* withTraceScope(
            {
              metadata: { callType: 'chat', forkId, forkName: agentInstance?.name ?? 'root', turnId, chainId },
              strategyId: 'xml-act',
              systemPrompt,
            },
            boundModel.invoke(
              CodingAgentChat,
              {
                systemPrompt,
                messages: chatMessages,
                options: { grammar: toolGrammar },
                ackTurns,
              },
              { inputTokenEstimate: compactionState.tokenEstimate },
            ),
          ).pipe(
            Effect.mapError((e) => TurnErrorCtor.LLMFailed({ message: e.message, cause: e })),
          )

          // Collect raw chunks and execute via xml-act runtime
          const xmlStream = cs.stream.pipe(
            Stream.tap(chunk => sink.emit({ _tag: 'RawResponseChunk', text: chunk })),
          )

          const executeResult = yield* execManager.execute(
            xmlStream,
            {
              forkId,
              turnId,
              chainId,
              defaultProseDest: agentDef.defaultRecipient,
              triggeredByUser,
              toolSet,
            },
            sink,
          )

          // Extract usage — tag as partial if execution failed
          const usage = cs.getUsage()
          const usageWithStatus = { ...usage, partial: executeResult.result._tag !== 'Completed' }

          // Offer final result
          yield* sink.emit({ _tag: 'TurnResult', value: { executeResult, usage: usageWithStatus } })
        }))

        // 3a. Drain turn stream, publishing events and collecting the final result
        const drained = yield* drainTurnEventStream(turnStream, forkId, turnId, publish)

        const { executeResult, usage } = drained.finalResult

        const inputTokens = usage.inputTokens
        if (inputTokens !== null) {
          logger.info({ inputTokens, usage, forkId, turnId }, '[Cortex] Captured usage from LLM response')
        }

        // 4. Detect output truncation — if output tokens hit the model limit,
        // the response was cut short (incomplete XML, broken tool calls, etc.)
        const outputTruncated = usage.outputTokens !== null
          && boundModel.model.maxOutputTokens !== null
          && usage.outputTokens >= boundModel.model.maxOutputTokens

        const turnResult: TurnOutcome = outputTruncated
          ? { _tag: 'SystemError', message: `Your response was truncated because it hit the maximum output token limit (${boundModel.model.maxOutputTokens} tokens). You MUST split your work into smaller steps — make fewer tool calls per turn, write shorter files, or break large operations into multiple turns.` }
          : executeResult.result

        // Publish turn_completed for this fork
        yield* publish({
          type: 'turn_completed',
          forkId,
          turnId,
          chainId,
          strategyId: 'xml-act',
          result: turnResult,
          inputTokens,
          outputTokens: usage.outputTokens,
          cacheReadTokens: usage.cacheReadTokens,
          cacheWriteTokens: usage.cacheWriteTokens,
          providerId: boundModel.model.providerId,
          modelId: boundModel.model.id,
        })


      }).pipe(
        Effect.onInterrupt(() => Effect.gen(function* () {
          const turnCompleted = yield* buildInterruptedTurnCompleted({ forkId, turnId, chainId })
          yield* publish(turnCompleted)
        }).pipe(Effect.orDie)),
        Effect.catchAll((error: TurnEngineCrash | TurnError) => Effect.gen(function* () {
          // TurnEngineCrash is an infrastructure defect — should not happen in normal operation
          if (error._tag === 'TurnEngineCrash') {
            // Improvement C: if crash was caused by a typed ModelError, surface it as a turn error
            const cause = error.cause
            if (isTaggedCause(cause)) {
              const errorType = cause._tag
              const errorMessage = cause.message ?? error.message
              logger.error({ context: 'Cortex', forkId, turnId, errorType }, `Cortex: Mid-stream ModelError (${errorType}): ${errorMessage}`)

              let message: string
              if (errorType === 'AuthFailed') {
                message = authReconnectMessage()
              } else if (errorType === 'ProviderDisconnected' && isAuthReconnectMessage(errorMessage)) {
                message = errorMessage
              } else if (errorType === 'SubscriptionRequired') {
                message = errorMessage
              } else if (errorType === 'UsageLimitExceeded') {
                message = errorMessage
              } else {
                message = `Stream error (${errorType}): ${errorMessage}`
              }

              yield* publish({
                type: 'turn_unexpected_error',
                forkId,
                turnId,
                message,
                errorCode: 'code' in cause ? (cause as any).code : undefined,
              })
              return
            }
            return yield* Effect.die(error)
          }

          const errorMessage = error.message
          const errorCause = error.cause

          // Tier 1: Definitive context-limit error (known provider patterns)
          const definiteContextLimit = classifyRetryability(errorCause) === 'context-limit'

          // Tier 2: Heuristic — if over soft cap and compacting, any error is likely context-related
          let probableContextLimit = false
          if (!definiteContextLimit) {
            const compactionState = yield* read(CompactionProjection, forkId)
            const ambientService = yield* AmbientServiceTag
            const configState = ambientService.getValue(ConfigAmbient)
            const heuristicAgentStatus = yield* read(AgentStatusProjection)
            const heuristicForkInfo = getForkInfo(heuristicAgentStatus, forkId)
            if (heuristicForkInfo) {
              const { softCap } = getSlotConfig(configState, heuristicForkInfo.slot)
              probableContextLimit = compactionState.tokenEstimate >= softCap && compactionState._tag !== 'idle'
            }
          }

          if (definiteContextLimit || probableContextLimit) {
            logger.warn({
              context: 'Cortex',
              forkId,
              turnId,
              chainId,
              definite: definiteContextLimit,
              probable: probableContextLimit,
              error: errorMessage
            }, 'Cortex: Context limit hit, blocking until compaction completes')

            yield* publish({
              type: 'context_limit_hit',
              forkId,
              error: errorMessage,
            })

            yield* publish({
              type: 'turn_completed',
              forkId,
              turnId,
              chainId,
              strategyId: 'xml-act',
              result: { _tag: 'SystemError', message: `Context limit exceeded, waiting for compaction: ${errorMessage}` },
              inputTokens: null,
              outputTokens: null,
              cacheReadTokens: null,
              cacheWriteTokens: null,
              providerId: resolvedProviderId,
              modelId: resolvedModelId,
            })
          } else if (classifyRetryability(errorCause) === 'auth') {
            yield* publish({
              type: 'turn_unexpected_error',
              forkId,
              turnId,
              message: authReconnectMessage(),
            })
          } else {
            logger.error({
              context: 'Cortex',
              forkId,
              turnId,
              chainId,
              error: errorMessage
            }, 'Cortex: LLM stream failed after all retries')

            const { message, errorCode } = buildGeneralErrorPayload(errorMessage, errorCause)

            yield* publish({
              type: 'turn_unexpected_error',
              forkId,
              turnId,
              message,
              errorCode,
            })
          }
        }))
      )
    }
  }
})