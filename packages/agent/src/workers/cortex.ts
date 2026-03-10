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

import { Effect, Stream, Queue, Schedule, Duration, Scope } from 'effect'
import { Worker, type PublishFn } from '@magnitudedev/event-core'
import type { XmlRuntimeCrash } from '@magnitudedev/xml-act'
import { actionsTagClose, TURN_CONTROL_NEXT, TURN_CONTROL_YIELD } from '@magnitudedev/xml-act'
import { logger } from '@magnitudedev/logger'
import { emitTrace } from '@magnitudedev/tracing'
import { isTracing } from '@magnitudedev/tracing'
import { isContextLimitError } from '../util/context-limit-error'
import type { ChatMessage } from '@magnitudedev/llm-core'
import { BamlClientHttpError, BamlValidationError } from '@magnitudedev/llm-core'
import { Image as BamlImage } from '@boundaryml/baml'
import type { ObservationPart } from '@magnitudedev/agent-definition'
import { getXmlActProtocol, buildAckTurn } from '@magnitudedev/agent-definition'
import subagentBasePrompt from '../agents/prompts/subagent-base.txt' with { type: 'text' }
import { ContentPart } from '../content'
import type { AppEvent, ResponsePart } from '../events'

import type { TurnEvent, TurnError, TurnStrategyResult } from '../execution/types'
import { createTurnStream, TurnError as TurnErrorCtor } from '../execution/types'
import { MemoryProjection, getView } from '../projections/memory'

import { LLMMessage } from '../projections/memory'
import { CompactionProjection } from '../projections/compaction'
import { SessionContextProjection } from '../projections/session-context'
import { ForkProjection } from '../projections/fork'
import { ExecutionManager } from '../execution/execution-manager'
import { getAgentDefinition, type AgentVariant } from '../agents'
import { generateXmlActToolDocs } from '../tools/xml-tool-docs'
import { getContextLimits } from '../constants'
import { resolveModel, detectProviders, getProvider, ensureValidAuth, primary, secondary, browser as browserProxy } from '@magnitudedev/providers'


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

function isNonRetryableError(error: unknown): boolean {
  if (error instanceof BamlClientHttpError) {
    const statusCode = error.status_code
    if (statusCode !== undefined && statusCode >= 400 && statusCode < 500) {
      if (statusCode !== 408 && statusCode !== 429) {
        logger.warn(`[Cortex] Non-retryable HTTP error (${statusCode}): ${error.message}`)
        return true
      }
    }
    return false
  }
  if (error instanceof BamlValidationError) {
    logger.warn(`[Cortex] Non-retryable validation error: ${error.message}`)
    return true
  }
  return false
}

const retrySchedule = Schedule.exponential(Duration.seconds(1), 1.5).pipe(
  Schedule.jittered,
  Schedule.intersect(Schedule.recurs(6))
)

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
// Turn Stream Consumption
// =============================================================================

/**
 * Drain a turn event stream, publishing each event to the bus.
 * Returns the final TurnStrategyResult.
 */
function drainTurnStream<R>(
  turnStream: Stream.Stream<TurnEvent, XmlRuntimeCrash | TurnError, R | Scope.Scope>,
  forkId: string | null,
  turnId: string,
  publish: PublishFn<AppEvent>,
): Effect.Effect<{ finalResult: TurnStrategyResult }, XmlRuntimeCrash | TurnError, R> {
  return Effect.gen(function* () {
    let finalResult: TurnStrategyResult | null = null

    yield* Effect.scoped(turnStream.pipe(
      Stream.runForEach((event) => Effect.gen(function* () {
        switch (event._tag) {
          case 'MessageStart':
            yield* publish({ type: 'message_start', forkId, turnId, id: event.id, dest: event.dest })
            break
          case 'MessageChunk':
            yield* publish({ type: 'message_chunk', forkId, turnId, id: event.id, text: event.text })
            break
          case 'MessageEnd':
            yield* publish({ type: 'message_end', forkId, turnId, id: event.id })
            break
          case 'ThinkingDelta':
            yield* publish({ type: 'thinking_chunk', forkId, turnId, text: event.text })
            break
          case 'ThinkingEnd':
            yield* publish({ type: 'thinking_end', forkId, turnId, about: event.about })
            break
          case 'LensStarted':
            yield* publish({ type: 'lens_start', forkId, turnId, name: event.name })
            break
          case 'LensDelta':
            yield* publish({ type: 'lens_chunk', forkId, turnId, text: event.text })
            break
          case 'LensEnded':
            yield* publish({ type: 'lens_end', forkId, turnId, name: event.name })
            break

          case 'ToolEvent':
            yield* publish({ type: 'tool_event', forkId, turnId, toolCallId: event.toolCallId, toolKey: event.toolKey, event: event.event, display: event.display })
            break
          case 'Trace':
            emitTrace(event.ctx, event.request, event.response, event.usage)
            break
          case 'TurnResult':
            finalResult = event.value
            break
        }
      }))
    ))

    if (!finalResult) {
      return yield* Effect.die(new Error('Turn stream ended without TurnResult'))
    }

    return { finalResult }
  })
}

// =============================================================================
// Worker
// =============================================================================

export const Cortex = Worker.defineForked<AppEvent>()({
  name: 'Cortex',

  forkLifecycle: {
    activateOn: 'fork_started',
    completeOn: 'fork_completed'
  },

  eventHandlers: {
    turn_started: (event, publish, read) => {
      const { forkId, turnId, chainId } = event

      const rawCodeChunks: string[] = []
      return Effect.gen(function* () {
        // 1. Get memory context for THIS fork (read auto-resolves forkId)
        const forkMemory = yield* read(MemoryProjection)
        const sessionCtx = yield* read(SessionContextProjection)
        const forkState = yield* read(ForkProjection)
        const forkInstance = forkId ? forkState.forks.get(forkId) : null

        // Determine agent: child forks use their role, root fork is always orchestrator.
        const variant: AgentVariant = forkInstance
          ? forkInstance.role as AgentVariant
          : 'orchestrator'

        const agentDef = getAgentDefinition(variant)
        const modelSlot = agentDef.model
        const timezone = sessionCtx.context?.timezone ?? null

        // Validate provider is connected before attempting LLM call
        const resolved = resolveModel(modelSlot)
        if (!resolved) {
          yield* publish({
            type: 'turn_unexpected_error',
            forkId,
            turnId,
            message: 'No model configured. Please connect a provider and select a model in /settings.',
          })
          return
        }

        const connectedProviderIds = new Set(detectProviders().map(d => d.provider.id))
        if (!connectedProviderIds.has(resolved.providerId)) {
          const providerName = getProvider(resolved.providerId)?.name ?? resolved.providerId
          yield* publish({
            type: 'turn_unexpected_error',
            forkId,
            turnId,
            message: `${providerName} is not connected. Please connect the provider or choose another provider/model in /settings.`,
          })
          return
        }

        // Build messages array
        const chatMessages: ChatMessage[] = toBamlMessages(getView(forkMemory.messages, timezone, 'agent'))

        // Run agent observables
        const execManager = yield* ExecutionManager
        const observations: ObservationPart[] = []
        if (forkId) {
          const boundObs = execManager.getObservables(forkId)
          for (const obs of boundObs) {
            const parts = yield* obs.observe()
            observations.push(...parts)
          }
        }

        // 2. Build system prompt with runtime protocol/tool-doc substitution
        const implicitTools = ['think'] as const
        const systemPrompt = variant === 'orchestrator'
          ? buildXmlActSystemPrompt(agentDef.systemPrompt, agentDef, implicitTools, 'user', 'orchestrator')
          : buildXmlActSystemPrompt(agentDef.systemPrompt, agentDef, implicitTools, 'parent', 'subagent')

        logger.info({ variant, forkId, turnId }, '[Cortex] Executing turn via xml-act')

        // 3. Build and consume the turn event stream
        const turnStream = createTurnStream((queue) => Effect.gen(function* () {
          const slot = modelSlot
          const startTime = Date.now()

          // Ensure OAuth token is fresh
          yield* Effect.tryPromise({
            try: () => ensureValidAuth(modelSlot),
            catch: (e) => TurnErrorCtor.AuthFailed({ message: e instanceof Error ? e.message : String(e), cause: e })
          })

          // Inject observations as user message
          if (observations.length > 0) {
            const observationContent: (string | BamlImage)[] = observations.map(part => {
              if (part.type === 'text') return part.text
              return BamlImage.fromBase64(part.mediaType, part.base64)
            })
            chatMessages.push({ role: 'user', content: observationContent })
          }

          // Stream BAML response with first-chunk retry
          let attempt = 0
          const { firstChunk, restOfStream, chatStream } = yield* Effect.retry(
            Effect.tryPromise({
              try: async () => {
                attempt++
                const model = modelSlot === 'primary' ? primary : modelSlot === 'browser' ? browserProxy : secondary
                const ackTurn = buildAckTurn(agentDef.thinkingLenses)
                const cs = model.chat(systemPrompt, chatMessages, { forkId, forkName: forkInstance?.name ?? 'root', turnId, chainId, callType: 'chat' }, {}, ackTurn)
                const iterator = cs.stream[Symbol.asyncIterator]()

                const firstResult = await iterator.next()

                if (firstResult.done) {
                  return { firstChunk: null, restOfStream: iterator, chatStream: cs }
                }
                return { firstChunk: firstResult.value, restOfStream: iterator, chatStream: cs }
              },
              catch: (e) => TurnErrorCtor.LLMFailed({ message: e instanceof Error ? e.message : String(e), cause: e })
            }),
            {
              schedule: retrySchedule,
              while: (error) => {
                const shouldNotRetry = isNonRetryableError(error.cause)
                logger.warn(`[Cortex] BAML stream connection failed (attempt ${attempt}/7): ${error.message}${shouldNotRetry ? '' : ' - will retry'}`)
                return !shouldNotRetry
              }
            }
          )

          // Combine first chunk with rest of stream
          async function* combinedStream() {
            try {
              if (firstChunk !== null) {
                yield firstChunk
              }
              let next = await restOfStream.next()
              while (!next.done) {
                yield next.value
                next = await restOfStream.next()
              }
            } finally {
              await restOfStream.return?.(undefined as never)
            }
          }

          const rawStream = Stream.fromAsyncIterable(
            combinedStream(),
            (e) => e instanceof Error ? e : new Error(String(e))
          )

          // Guard the stream first (truncate after \n</actions>, inject if missing),
          // then tap to accumulate raw XML chunks (for interrupt preservation).
          // Guard must come before the tap so injected closing tags are captured.
          const xmlStream = rawStream.pipe(
            Stream.tap(chunk => Effect.sync(() => { rawCodeChunks.push(chunk) }))
          )

          // Execute via xml-act runtime
          const executeResult = yield* execManager.execute(
            xmlStream,
            { forkId, turnId, chainId },
            queue,
          )

          // Extract usage
          const usage = chatStream.getUsage()

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

          // Emit trace
          if (isTracing()) {
            const resolvedModel = resolveModel(slot)
            const collectorData = chatStream.getCollectorData()
            const rawBody = collectorData.rawRequestBody
            const collectorMessages = rawBody != null && typeof rawBody === 'object' && 'messages' in rawBody && Array.isArray(rawBody.messages)
              ? rawBody.messages
              : null

            yield* Queue.offer(queue, {
              _tag: 'Trace',
              ctx: {
                startTime,
                model: resolvedModel?.modelId ?? null,
                provider: resolvedModel?.providerId ?? null,
                slot,
                defaultCallType: 'chat',
                meta: { callType: 'chat', forkId, forkName: forkInstance?.name ?? 'root', turnId, chainId },
                strategyId: 'xml-act',
                systemPrompt,
              },
              request: { messages: collectorMessages ?? chatMessages },
              response: {
                rawBody: collectorData.rawResponseBody,
                sseEvents: collectorData.sseEvents,
                rawOutput: rawCode,
              },
              usage,
            })
          }

          // Offer final result
          yield* Queue.offer(queue, { _tag: 'TurnResult', value: { executeResult, usage, responseParts, rawCodeChunks } })
        }))

        // 3a. Drain turn stream, publishing events and collecting the final result
        const drained = yield* drainTurnStream(turnStream, forkId, turnId, publish)

        const { executeResult, usage, responseParts } = drained.finalResult

        const inputTokens = usage.inputTokens
        if (inputTokens !== null) {
          logger.info({ inputTokens, usage, forkId, turnId }, '[Cortex] Captured usage from LLM response')
        }

        // 4. Detect output truncation — if output tokens hit the model limit,
        // the response was cut short (incomplete XML, broken tool calls, etc.)
        const outputTruncated = usage.outputTokens !== null
          && resolved.maxOutputTokens !== undefined
          && usage.outputTokens >= resolved.maxOutputTokens

        const turnResult = outputTruncated
          ? { success: false as const, error: `Your response was truncated because it hit the maximum output token limit (${resolved.maxOutputTokens} tokens). You MUST split your work into smaller steps — make fewer tool calls per turn, write shorter files, or break large operations into multiple turns.`, cancelled: false }
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
        Effect.onInterrupt(() => {
          const rawCode = rawCodeChunks.join('')
          const interruptResponseParts: readonly ResponsePart[] = rawCode.trim()
            ? [{ type: 'text', content: rawCode }]
            : []
          return publish({
            type: 'turn_completed',
            forkId,
            turnId,
            chainId,
            strategyId: 'xml-act',
            responseParts: interruptResponseParts,
            toolCalls: [],
            inspectResults: [],
            result: { success: false, error: 'Interrupted', cancelled: true },
            inputTokens: null,
            outputTokens: null,
            cacheReadTokens: null,
            cacheWriteTokens: null,
          })
        }),
        Effect.catchAll((error: XmlRuntimeCrash | TurnError) => Effect.gen(function* () {
          // XmlRuntimeCrash is an infrastructure defect — should not happen in normal operation
          if (error._tag === 'XmlRuntimeCrash') {
            return yield* Effect.die(error)
          }

          const errorMessage = error.message
          const errorCause = error.cause

          // Tier 1: Definitive context-limit error (known provider patterns)
          const definiteContextLimit = isContextLimitError(errorCause)

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
