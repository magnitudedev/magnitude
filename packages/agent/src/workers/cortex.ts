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

import { Effect, Stream, Queue, Either } from 'effect'
import { Worker } from '@magnitudedev/event-core'
import { actionsTagClose, TURN_CONTROL_NEXT, TURN_CONTROL_YIELD } from '@magnitudedev/xml-act'
import type { XmlRuntimeCrash } from '@magnitudedev/xml-act'
import { logger } from '@magnitudedev/logger'

import { ContextLimitExceeded, AuthFailed, TransportError as ProviderTransportError, ParseError as ProviderParseError, NotConfigured, ProviderDisconnected } from '@magnitudedev/providers'
import type { ModelError } from '@magnitudedev/providers'
import { drainTurnEventStream } from './turn-event-drain'
import type { ChatMessage } from '@magnitudedev/llm-core'
import { BamlClientHttpError, BamlValidationError } from '@magnitudedev/llm-core'
import { Image as BamlImage } from '@boundaryml/baml'
import type { ObservationPart } from '@magnitudedev/agent-definition'
import { getXmlActProtocol, buildAckTurn } from '@magnitudedev/agent-definition'
import subagentBasePrompt from '../agents/prompts/subagent-base.txt' with { type: 'text' }
import { ContentPart } from '../content'
import type { AppEvent, ResponsePart } from '../events'

import { createTurnStream, TurnError as TurnErrorCtor } from '../execution/types'
import type { TurnError } from '../execution/types'
import { MemoryProjection, getView } from '../projections/memory'

import { LLMMessage } from '../projections/memory'
import { CompactionProjection } from '../projections/compaction'
import { SessionContextProjection } from '../projections/session-context'
import { AgentRoutingProjection, getAgentByForkId } from '../projections/agent-routing'
import { WorkingStateProjection } from '../projections/working-state'
import { ExecutionManager } from '../execution/execution-manager'
import { getAgentDefinition, type AgentVariant } from '../agents'
import { generateXmlActToolDocs } from '../tools/xml-tool-docs'
import { getContextLimits } from '../constants'
import { ModelResolver, CodingAgentChat } from '@magnitudedev/providers'
import { withTraceScope } from '../tracing'
import { buildInterruptedTurnCompleted } from '../util/interrupt-utils'


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
// Error Classification
// =============================================================================

type NonRetryableReason = 'context-limit' | 'auth' | 'parse' | 'client-error' | 'not-configured' | 'disconnected' | null

function classifyRetryability(error: unknown): NonRetryableReason {
  if (error instanceof ContextLimitExceeded) return 'context-limit'
  if (error instanceof AuthFailed) return 'auth'
  if (error instanceof ProviderParseError) return 'parse'
  if (error instanceof ProviderTransportError) {
    const s = error.status
    if (s !== null && s >= 400 && s < 500 && s !== 408 && s !== 429) return 'client-error'
    return null
  }
  // Legacy BAML error fallback
  if (error instanceof BamlClientHttpError) {
    const s = error.status_code
    if (s !== undefined && s >= 400 && s < 500 && s !== 408 && s !== 429) return 'client-error'
    return null
  }
  if (error instanceof BamlValidationError) return 'parse'
  return null
}


// =============================================================================
// System Prompt Builder
// =============================================================================

function buildXmlActSystemPrompt(
  roleDescription: string,
  agentDef: ReturnType<typeof getAgentDefinition>,
  implicitTools: readonly string[] = ['think'],
  defaultRecipient = 'user',
  role: 'orchestrator' | 'subagent' = 'orchestrator',
): string {
  const toolDocs = generateXmlActToolDocs(agentDef, implicitTools)
  return roleDescription
    .replaceAll('{{RESPONSE_PROTOCOL}}', getXmlActProtocol(defaultRecipient, agentDef.thinkingLenses, role))
    .replaceAll('{{TOOL_DOCS}}', toolDocs)
    .replaceAll('{{SUBAGENT_BASE}}', subagentBasePrompt)
}

// =============================================================================
// Worker
// =============================================================================

export const Cortex = Worker.defineForked<AppEvent>()({
  name: 'Cortex',

  forkLifecycle: {
    activateOn: 'agent_created',
    completeOn: 'agent_dismissed'
  },

  eventHandlers: {
    turn_started: (event, publish, read) => {
      const { forkId, turnId, chainId } = event

      const rawCodeChunks: string[] = []
      return Effect.gen(function* () {
        const sessionCtx = yield* read(SessionContextProjection)
        const agentState = yield* read(AgentRoutingProjection)
        const agentInstance = forkId ? getAgentByForkId(agentState, forkId) : null

        // Determine agent: child forks use their role, root fork is always orchestrator.
        const variant: AgentVariant = agentInstance
          ? agentInstance.role as AgentVariant
          : 'orchestrator'

        const agentDef = getAgentDefinition(variant)
        const modelSlot = agentDef.model
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

        // Build messages array (now includes observations in system inbox)
        const forkMemory = yield* read(MemoryProjection)
        const chatMessages: ChatMessage[] = toBamlMessages(getView(forkMemory.messages, timezone, 'agent'))

        // 2. Build system prompt with runtime protocol/tool-doc substitution
        const implicitTools = ['think'] as const
        const systemPrompt = variant === 'orchestrator'
          ? buildXmlActSystemPrompt(agentDef.systemPrompt, agentDef, implicitTools, 'user', 'orchestrator')
          : buildXmlActSystemPrompt(agentDef.systemPrompt, agentDef, implicitTools, 'parent', 'subagent')

        logger.info({ variant, forkId, turnId }, '[Cortex] Executing turn via xml-act')

        // Step 1: Ensure model ready
        const resolveResult = yield* runtime.resolve(modelSlot).pipe(Effect.either)
        if (Either.isLeft(resolveResult)) {
          const e = resolveResult.left
          const message = e._tag === 'NotConfigured'
            ? 'No model configured. Please connect a provider and select a model in /settings.'
            : e._tag === 'ProviderDisconnected'
            ? e.message
            : `Authentication failed: ${e.message}`
          yield* publish({ type: 'turn_unexpected_error', forkId, turnId, message })
          return
        }
        const boundModel = resolveResult.right

        // 3. Build and consume the turn event stream
        const turnStream = createTurnStream((queue) => Effect.gen(function* () {

          const ackTurn = buildAckTurn(agentDef.thinkingLenses)
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
                options: {},
                ackTurn,
              },
            ),
          ).pipe(
            Effect.mapError((e) => TurnErrorCtor.LLMFailed({ message: e.message, cause: e })),
          )

          // Collect raw chunks and execute via xml-act runtime
          const xmlStream = cs.stream.pipe(
            Stream.tap(chunk => Effect.sync(() => { rawCodeChunks.push(chunk) })),
          )

          const executeResult = yield* execManager.execute(
            xmlStream,
            { forkId, turnId, chainId },
            queue,
          )

          // Extract usage — tag as partial if execution failed
          const usage = cs.getUsage()
          const usageWithStatus = { ...usage, partial: executeResult.result.success === false }

          // Build response parts — single text part containing the raw XML
          // If there's a synthetic inspect block, splice it inside the last </actions> tag
          // so the LLM memory sees it as part of the actions block
          let rawCode = rawCodeChunks.join('')

          if (executeResult.syntheticInspectCode) {
            const idx = rawCode.lastIndexOf(actionsTagClose())
            if (idx !== -1) {
              rawCode = rawCode.slice(0, idx) + executeResult.syntheticInspectCode + rawCode.slice(idx)
            }
          }
          const responseParts: readonly ResponsePart[] = rawCode.trim()
            ? [{ type: 'text', content: rawCode }]
            : []

          // Offer final result
          yield* Queue.offer(queue, { _tag: 'TurnResult', value: { executeResult, usage: usageWithStatus, responseParts, rawCodeChunks } })
        }))

        // 3a. Drain turn stream, publishing events and collecting the final result
        const drained = yield* drainTurnEventStream(turnStream, forkId, turnId, publish)

        const { executeResult, usage, responseParts } = drained.finalResult

        const inputTokens = usage.inputTokens
        if (inputTokens !== null) {
          logger.info({ inputTokens, usage, forkId, turnId }, '[Cortex] Captured usage from LLM response')
        }

        // 4. Detect output truncation — if output tokens hit the model limit,
        // the response was cut short (incomplete XML, broken tool calls, etc.)
        const outputTruncated = usage.outputTokens !== null
          && boundModel.model.maxOutputTokens !== null
          && usage.outputTokens >= boundModel.model.maxOutputTokens

        const turnResult = outputTruncated
          ? { success: false as const, error: `Your response was truncated because it hit the maximum output token limit (${boundModel.model.maxOutputTokens} tokens). You MUST split your work into smaller steps — make fewer tool calls per turn, write shorter files, or break large operations into multiple turns.`, cancelled: false }
          : executeResult.result

        // Publish turn_completed for this fork
        yield* publish({
          type: 'turn_completed',
          forkId,
          turnId,
          chainId,
          strategyId: 'xml-act',
          responseParts,
          toolCalls: executeResult.toolCalls,
          inspectResults: executeResult.inspectResults,
          result: turnResult,
          inputTokens,
          outputTokens: usage.outputTokens,
          cacheReadTokens: usage.cacheReadTokens,
          cacheWriteTokens: usage.cacheWriteTokens,
        })


      }).pipe(
        Effect.onInterrupt(() => Effect.gen(function* () {
          const forkWorkingState = yield* read(WorkingStateProjection)

          if (forkWorkingState.working === false) {
            return
          }

          const turnCompleted = yield* buildInterruptedTurnCompleted({ forkId, turnId, chainId })
          yield* publish(turnCompleted)
        }).pipe(Effect.orDie)),
        Effect.catchAll((error: XmlRuntimeCrash | TurnError) => Effect.gen(function* () {
          // XmlRuntimeCrash is an infrastructure defect — should not happen in normal operation
          if (error._tag === 'XmlRuntimeCrash') {
            // Improvement C: if crash was caused by a typed ModelError, surface it as a turn error
            const cause = error.cause
            if (cause && typeof cause === 'object' && '_tag' in cause) {
              const errorType = (cause as any)._tag as string
              const errorMessage = (cause as any).message as string ?? error.message
              logger.error({ context: 'Cortex', forkId, turnId, errorType }, `Cortex: Mid-stream ModelError (${errorType}): ${errorMessage}`)
              yield* publish({
                type: 'turn_unexpected_error',
                forkId,
                turnId,
                message: `Stream error (${errorType}): ${errorMessage}`,
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
            const compactionState = yield* read(CompactionProjection)
            const { softCap } = getContextLimits()
            probableContextLimit = compactionState.tokenEstimate >= softCap && (compactionState.isCompacting || compactionState.pendingFinalization)
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
              responseParts: [],
              toolCalls: [],
              inspectResults: [],
              result: { success: false, error: `Context limit exceeded, waiting for compaction: ${errorMessage}`, cancelled: false },
              inputTokens: null,
              outputTokens: null,
              cacheReadTokens: null,
              cacheWriteTokens: null,
            })
          } else {
            logger.error({
              context: 'Cortex',
              forkId,
              turnId,
              chainId,
              error: errorMessage
            }, 'Cortex: LLM stream failed after all retries')

            yield* publish({
              type: 'turn_unexpected_error',
              forkId,
              turnId,
              message: `Unexpected error while executing turn: ${errorMessage}`
            })
          }
        }))
      )
    }
  }
})