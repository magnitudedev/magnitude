/**
 * CompactionWorker (Forked)
 *
 * Handles conversation history compaction to prevent token overflow.
 * Per-fork worker that responds to signals and events for compaction lifecycle:
 *
 * 1. shouldCompactChanged signal → start BAML summarization, publish compaction_ready
 * 2. turn_completed event → if pending compaction and not working, finalize:
 *    capture vars, reset sandbox, publish compaction_completed + sandbox_reset
 *
 * Design decisions:
 * - Turns are NOT blocked during BAML summarization (between compaction_started and compaction_ready)
 * - Turns ARE blocked after compaction_ready until compaction_completed (via shouldTrigger gate)
 * - Context limit blocking: proactive (estimate >= hardCap during compaction) + reactive (context_limit_hit event)
 *
 * All compaction data is event-sourced via compaction_ready → MemoryProjection.
 * No local state needed — the worker reads pending data from the projection.
 */

import { Effect } from 'effect'
import { Worker, type PublishFn, type WorkerReadFn } from '@magnitudedev/event-core'
import { logger } from '@magnitudedev/logger'
import { type ChatMessage } from '@magnitudedev/llm-core'
import { Image as BamlImage } from '@boundaryml/baml'
import type { AppEvent } from '../events'
import { textOf, ContentPart } from '../content'
import { MemoryProjection, getView, LLMMessage, type ForkMemoryState } from '../projections/memory'
import { CompactionProjection } from '../projections/compaction'
import type { ForkCompactionState } from '../projections/compaction'
import { SessionContextProjection } from '../projections/session-context'
import { WorkingStateProjection } from '../projections/working-state'
// ExecutionManager no longer needed — xml-act is stateless, no sandbox reset
import { KEEP_MESSAGE_RATIO, CHARS_PER_TOKEN, EMERGENCY_COMPACT_CONTEXT_TRIM_RATIO } from '../constants'
import { isContextLimitError } from '../util/context-limit-error'
import { primary } from '@magnitudedev/providers'
import { getAgentDefinition } from '../agents'
import { getContextLimits } from '../constants'
// compactionVariableNote removed — xml-act has no cross-turn variables
import { collectSessionContext } from '../util/collect-session-context'

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
export const CompactionWorker = Worker.defineForked<AppEvent>()({
  name: 'CompactionWorker',

  forkLifecycle: {
    activateOn: 'fork_started',
    completeOn: 'fork_completed'
  },

  eventHandlers: {
    // Finalize pending compaction when a turn ends and sandbox is idle
    turn_completed: (event, publish, read) => Effect.gen(function* () {
      const { forkId } = event

      const compactionState: ForkCompactionState = yield* read(CompactionProjection)
      if (!compactionState.pendingFinalization || !compactionState.pendingCompactionData) return

      yield* finalizeCompaction(forkId, publish, read)
    })
  },

  signalHandlers: (on) => [
    // Phase 1: Start BAML summarization when memory exceeds budget
    on(CompactionProjection.signals.shouldCompactChanged, ({ forkId, shouldCompact }, publish, read) =>
      Effect.gen(function* () {
        if (!shouldCompact) return

        const forkMemory: ForkMemoryState = yield* read(MemoryProjection)
        const compactionState: ForkCompactionState = yield* read(CompactionProjection)
        const sessionCtx = yield* read(SessionContextProjection)
        const timezone = sessionCtx.context?.timezone ?? null

        // Skip if already compacting or pending finalization
        if (compactionState.isCompacting || compactionState.pendingFinalization) return

        // Calculate how many messages to compact
        const keepTokenBudget = getContextLimits().softCap * KEEP_MESSAGE_RATIO
        let keepTokens = 0
        let keepCount = 0

        for (let i = forkMemory.messages.length - 1; i >= 1; i--) {
          const llmMessage = getView([forkMemory.messages[i]], timezone, 'agent')[0]
          if (!llmMessage) continue
          const msgTokens = Math.ceil(textOf(llmMessage.content).length / CHARS_PER_TOKEN)
          if (keepTokens + msgTokens > keepTokenBudget) break
          keepTokens += msgTokens
          keepCount++
        }

        const compactedMessageCount = Math.max(0, forkMemory.messages.length - 1 - keepCount)
        if (compactedMessageCount === 0) {
          logger.info('[CompactionWorker] Not enough messages to compact')
          return
        }

        logger.info({ compactedMessageCount, tokenEstimate: compactionState.tokenEstimate, messageCount: forkMemory.messages.length }, '[CompactionWorker] Starting compaction')

        yield* publish({
          type: 'compaction_started',
          forkId,
          compactedMessageCount,
        })

        // Get messages to compact
        const messagesToCompact = forkMemory.messages.slice(1, 1 + compactedMessageCount)
        const llmMessages = getView(messagesToCompact, timezone, 'agent')
        const chatMessages = toBamlMessages(llmMessages)

        // Calculate original token estimate for tokensSaved computation later
        const originalTokenEstimate = chatMessages.reduce(
          (sum, msg) => sum + Math.ceil(JSON.stringify(msg).length / CHARS_PER_TOKEN), 0
        )

        // Run BAML summarization with trim-and-retry.
        // If the BAML call fails with a context-limit error, trim the oldest 20%
        // of messages from the input and retry. This ensures we get the best
        // summary possible from whatever messages fit.

        let trimmedMessages = chatMessages
        const maxRetries = 5

        const bamlEffect = Effect.gen(function* () {
          for (let attempt = 0; attempt <= maxRetries; attempt++) {
            try {
              const { result } = yield* Effect.tryPromise({
                try: () => primary.compact(getAgentDefinition('orchestrator').systemPrompt, trimmedMessages, { forkId, callType: 'compact' }),
                catch: (error) => error instanceof Error ? error : new Error(String(error))
              })
              return result
            } catch (error) {
              if (isContextLimitError(error) && trimmedMessages.length > 1 && attempt < maxRetries) {
                const trimAmount = Math.max(1, Math.floor(trimmedMessages.length * EMERGENCY_COMPACT_CONTEXT_TRIM_RATIO))
                logger.warn({ attempt, trimAmount, remaining: trimmedMessages.length - trimAmount }, '[CompactionWorker] Compaction BAML failed with context error, trimming and retrying')
                trimmedMessages = trimmedMessages.slice(trimAmount)
                continue
              }
              throw error
            }
          }
          logger.error('[CompactionWorker] Exhausted trim retries, using fallback truncation summary')
          return '[Previous conversation history was truncated because it exceeded the context window. The agent is continuing with recent messages only.]'
        })

        const contextEffect = Effect.tryPromise({
          try: () => collectSessionContext(),
          catch: (error) => error instanceof Error ? error : new Error(String(error))
        }).pipe(
          Effect.catchAll((error) => {
            logger.warn({ error: String(error) }, '[CompactionWorker] Failed to refresh session context')
            return Effect.succeed(null)
          })
        )

        yield* Effect.all([bamlEffect, contextEffect], { concurrency: 2 }).pipe(
          Effect.flatMap(([bamlSummary, refreshedContext]) => Effect.gen(function* () {
            // Publish compaction_ready — data is stored in MemoryProjection, gates shouldTrigger
            yield* publish({
              type: 'compaction_ready',
              forkId,
              summary: bamlSummary,
              compactedMessageCount,
              originalTokenEstimate,
              refreshedContext,
            })

            // Check if already between turns — if so, finalize immediately
            const workingState = yield* read(WorkingStateProjection)
            if (!workingState.working) {
              yield* finalizeCompaction(forkId, publish, read)
            }
            // Otherwise, turn_completed event handler will trigger finalization
          })),
          Effect.catchAllCause((cause) => Effect.gen(function* () {
            logger.error({ cause: cause.toString() }, '[CompactionWorker] Compaction failed')
            yield* publish({
              type: 'compaction_failed',
              forkId,
              error: cause.toString(),
            })
          }))
        )
      })
    )
  ]
})

/** Finalize a pending compaction: publish compaction_completed. No sandbox reset needed — xml-act is stateless. */
function finalizeCompaction(
  forkId: string | null,
  publish: PublishFn<AppEvent>,
  read: WorkerReadFn<AppEvent>
) {
  return Effect.gen(function* () {
    const compactionState: ForkCompactionState = yield* read(CompactionProjection)
    const pending = compactionState.pendingCompactionData
    if (!pending) return

    const summary = pending.summary

    const summaryTokens = Math.ceil(summary.length / CHARS_PER_TOKEN)
    const tokensSaved = pending.originalTokenEstimate - summaryTokens

    logger.info({ tokensSaved, compactedMessageCount: pending.compactedMessageCount }, '[CompactionWorker] Compaction completed')

    yield* publish({
      type: 'compaction_completed',
      forkId,
      summary,
      compactedMessageCount: pending.compactedMessageCount,
      tokensSaved,
      preservedVariables: [],
      refreshedContext: pending.refreshedContext,
    })
  }).pipe(
    Effect.catchAllCause((cause) => Effect.gen(function* () {
      logger.error({ cause: cause.toString() }, '[CompactionWorker] Finalization failed')
      yield* publish({
        type: 'compaction_failed',
        forkId,
        error: cause.toString(),
      })
    }))
  )
}
