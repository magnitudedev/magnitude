/**
 * HarnessEvent → AppEvent Adapter
 *
 * Translates harness-level events into agent app events.
 * Maintains per-turn state: message counters, tool tracking,
 * content fingerprinting (circuit breaker), and turn continuation policy.
 */

import { Effect } from 'effect'
import { logger } from '@magnitudedev/logger'
import type {
  HarnessEvent,
  ToolLifecycleEvent,
} from '@magnitudedev/harness'
import type { ProviderToolCallId } from '@magnitudedev/ai'
import {
  type AppEvent,
  type MessageDestination,
  type TurnOutcome as AgentTurnOutcome,
  type TurnCompletion,
  type TurnFeedback,
  type YieldTarget,
  outcomeWillChainContinue,
} from '../events'
import { finalizeModelAttemptFailure } from '../errors'

import type { ToolKey } from '../tools/toolkits'
import { toToolKeyErased } from '../tools/toolkits'
import type { RoleId } from '../agents/role-validation'
import type { ExecuteResult } from './types'
import { IDENTICAL_RESPONSE_BREAKER_THRESHOLD } from './types'


// ── Identical Response Tracker ───────────────────────────────────────

export interface IdenticalResponseTracker {
  lastResponseText: string
  consecutiveCount: number
}

// ── Config ───────────────────────────────────────────────────────────

export interface HarnessAdapterConfig {
  readonly forkId: string | null
  readonly turnId: string
  readonly chainId: string
  readonly roleId: RoleId
  readonly defaultProseDest: MessageDestination
  readonly publish: (event: AppEvent) => Effect.Effect<void>
  readonly identicalResponseTracker: IdenticalResponseTracker | null
  readonly retryCount: number
  readonly maxRetries: number
  /** Resolve a tool's model-facing name to the internal catalog key. */
  readonly resolveToolKey: (toolName: string) => ToolKey | undefined
}

// ── Adapter ──────────────────────────────────────────────────────────

export interface HarnessAdapter {
  readonly processEvent: (event: HarnessEvent) => Effect.Effect<void>
  readonly getResult: () => ExecuteResult
  readonly getIdenticalResponseTracker: () => IdenticalResponseTracker | null
}

export function createHarnessAdapter(config: HarnessAdapterConfig): HarnessAdapter {
  const {
    forkId,
    turnId,
    defaultProseDest,
    publish,
    resolveToolKey,
    retryCount,
    maxRetries,
  } = config

  // ── Per-turn mutable state ───────────────────────────────────────

  let messageCounter = 0
  let currentMessageId: string | null = null

  const toolsCalledKeys: ToolKey[] = []
  let lastToolKey: ToolKey | null = null
  let hasToolErrors = false
  let hasAnyResponseContent = false

  // toolCallId → ToolKey tracking
  const toolCallKeys = new Map<string, ToolKey>()

  // Content fingerprint for circuit breaker
  let contentFingerprint = ''

  const feedback: TurnFeedback[] = []
  let yieldTarget: YieldTarget | null = null

  // Result state
  let executionResult: AgentTurnOutcome = {
    _tag: 'Completed',
    completion: { toolCallsCount: 0, finishReason: 'stop', feedback: [], yieldTarget: null },
    requestId: null,
  }
  let turnUsage: ExecuteResult['usage'] = null
  let commitPolicy: ExecuteResult['commitPolicy'] = { _tag: 'commitCleanTurn' }

  // Circuit breaker state (mutated, returned via getter)
  let trackerState = config.identicalResponseTracker
    ? { ...config.identicalResponseTracker }
    : null

  // ── Helpers ──────────────────────────────────────────────────────

  const resolveDestination = (): MessageDestination => {
    return defaultProseDest
  }

  const emitToolEvent = (toolCallId: string, providerToolCallId: ProviderToolCallId, toolKey: ToolKey, event: ToolLifecycleEvent): Effect.Effect<void> =>
    publish({
      type: 'tool_event' as const,
      forkId,
      turnId,
      toolCallId,
      providerToolCallId,
      toolKey: toToolKeyErased(toolKey),
      event,
    })

  // ── Process ──────────────────────────────────────────────────────

  const processEvent = (event: HarnessEvent): Effect.Effect<void> =>
    Effect.gen(function* () {
      switch (event._tag) {
        // ── Thinking ─────────────────────────────────────────────
        case 'ThoughtStart': {
          hasAnyResponseContent = true
          yield* publish({
            type: 'thinking_start',
            forkId,
            turnId,
          })
          break
        }

        case 'ThoughtDelta': {
          contentFingerprint += event.text
          yield* publish({
            type: 'thinking_chunk',
            forkId,
            turnId,
            text: event.text,
          })
          break
        }

        case 'ThoughtEnd': {
          yield* publish({
            type: 'thinking_end',
            forkId,
            turnId,
          })
          break
        }

        // ── Messages ─────────────────────────────────────────────
        case 'MessageStart': {
          hasAnyResponseContent = true
          messageCounter++
          const messageId = `${turnId}-msg-${messageCounter}`
          currentMessageId = messageId

          const destination = resolveDestination()

          yield* publish({
            type: 'message_start',
            forkId,
            turnId,
            id: messageId,
            destination,
          })
          break
        }

        case 'MessageDelta': {
          if (currentMessageId === null) break
          contentFingerprint += event.text
          yield* publish({
            type: 'message_chunk',
            forkId,
            turnId,
            id: currentMessageId,
            text: event.text,
          })
          break
        }

        case 'MessageEnd': {
          if (currentMessageId === null) break
          yield* publish({
            type: 'message_end',
            forkId,
            turnId,
            id: currentMessageId,
          })
          currentMessageId = null
          break
        }

        // ── Tool Input Lifecycle ─────────────────────────────────
        case 'ToolInputStarted': {
          hasAnyResponseContent = true
          const toolKey = resolveToolKey(event.toolName)
          if (!toolKey) break
          toolCallKeys.set(event.toolCallId, toolKey)
          yield* emitToolEvent(event.toolCallId, event.providerToolCallId, toolKey, event)
          break
        }

        case 'ToolInputFieldChunk': {
          const toolKey = toolCallKeys.get(event.toolCallId)
          if (!toolKey) break
          contentFingerprint += event.delta
          yield* emitToolEvent(event.toolCallId, event.providerToolCallId, toolKey, event)
          break
        }

        case 'ToolInputFieldComplete': {
          const toolKey = toolCallKeys.get(event.toolCallId)
          if (!toolKey) break
          yield* emitToolEvent(event.toolCallId, event.providerToolCallId, toolKey, event)
          break
        }

        case 'ToolInputReady': {
          const toolKey = toolCallKeys.get(event.toolCallId)
          if (!toolKey) break
          yield* emitToolEvent(event.toolCallId, event.providerToolCallId, toolKey, event)
          break
        }

        case 'ToolInputRejected': {
          const toolKey = toolCallKeys.get(event.toolCallId)
          if (!toolKey) break
          yield* emitToolEvent(event.toolCallId, event.providerToolCallId, toolKey, event)
          break
        }

        // ── Tool Execution Lifecycle ─────────────────────────────
        case 'ToolExecutionStarted': {
          const toolKey = toolCallKeys.get(event.toolCallId)
          if (!toolKey) break
          yield* emitToolEvent(event.toolCallId, event.providerToolCallId, toolKey, event)
          break
        }

        case 'ToolEmission': {
          const toolKey = toolCallKeys.get(event.toolCallId)
          if (!toolKey) break
          yield* emitToolEvent(event.toolCallId, event.providerToolCallId, toolKey, event)
          break
        }

        case 'ToolExecutionEnded': {
          const toolKey = toolCallKeys.get(event.toolCallId)
          if (!toolKey) break

          // Track tool calls
          toolsCalledKeys.push(toolKey)
          lastToolKey = toolKey

          // Track errors
          if (event.result._tag === 'Error') {
            hasToolErrors = true
          }

          // Capture yield target from spawn/message worker tools
          if ((toolKey === 'spawnWorker' || toolKey === 'messageWorker') && event.result._tag === 'Success') {
            const output = event.result.output as Record<string, unknown>
            if (output.yield === true) {
              yieldTarget = 'workers'
            }
          }

          yield* emitToolEvent(event.toolCallId, event.providerToolCallId, toolKey, event)
          break
        }

        // ── Turn End ─────────────────────────────────────────────
        case 'TurnEnd': {
          const outcome = event.outcome
          const requestId = outcome.requestId

          // Capture usage
          if (event.usage) {
            turnUsage = {
              inputTokens: event.usage.inputTokens ?? null,
              outputTokens: event.usage.outputTokens ?? null,
              cacheReadTokens: event.usage.cacheReadTokens ?? null,
              cacheWriteTokens: event.usage.cacheWriteTokens ?? null,
              cost: event.usage.cost ?? null,
            }
          }

          const completed = (toolCallsCount: number): AgentTurnOutcome => ({
            _tag: 'Completed',
            completion: {
              toolCallsCount,
              finishReason: toolCallsCount > 0 ? 'tool_calls' : 'stop',
              feedback: [...feedback],
              yieldTarget,
            } satisfies TurnCompletion,
            requestId,
          })

          switch (outcome._tag) {
            case 'Completed': {
              let willContinue: boolean

              if (hasToolErrors || feedback.length > 0) {
                willContinue = true
              } else if (!hasAnyResponseContent) {
                willContinue = true
              } else {
                // Continue if any tools were called, stop otherwise
                willContinue = outcome.toolCallsCount > 0
              }

              executionResult = completed(willContinue ? Math.max(outcome.toolCallsCount, 1) : 0)
              break
            }

            case 'GateRejected': {
              executionResult = {
                _tag: 'GateRejected',
                toolCallId: outcome.toolCallId,
                providerToolCallId: outcome.providerToolCallId,
                toolName: outcome.toolName,
                requestId,
              }
              break
            }

            case 'ToolExecutionError': {
              // Tool execution failed — chain-continue so the model can respond to the error.
              executionResult = {
                _tag: 'ToolExecutionError',
                toolCallId: outcome.toolCallId,
                providerToolCallId: outcome.providerToolCallId,
                toolName: outcome.toolName,
                toolKey: outcome.toolKey,
                error: outcome.error,
                requestId,
              }
              break
            }

            case 'ToolInputValidationFailure': {
              // Streaming validation failed — chain-continue so the model can respond to the error.
              executionResult = {
                _tag: 'ToolInputValidationFailure',
                toolCallId: outcome.toolCallId,
                providerToolCallId: outcome.providerToolCallId,
                toolName: outcome.toolName,
                toolKey: outcome.toolKey,
                issue: outcome.issue,
                requestId,
              }
              break
            }

            case 'EngineDefect': {
              logger.error({ message: outcome.message }, '[HarnessAdapter] Engine defect')
              executionResult = {
                _tag: 'UnexpectedError',
                detail: { _tag: 'EngineDefect', message: outcome.message },
                requestId,
              }
              break
            }

            case 'StreamFailed': {
              const cause = outcome.terminal.cause
              const decision = finalizeModelAttemptFailure({ failure: cause, retryCount, maxRetries })
              logger.warn({
                message: outcome.message,
                retry: decision.retry._tag,
                streamFailure: decision.snapshot.tag,
                detailTag: decision.snapshot.detailTag,
              }, '[HarnessAdapter] Stream failed')

              executionResult = decision.outcome
              commitPolicy = decision.commitPolicy
              break
            }

            case 'OutputTruncated': {
              executionResult = { _tag: 'OutputTruncated', requestId }
              break
            }

            case 'ContentFiltered': {
              executionResult = {
                _tag: 'SafetyStop',
                reason: { _tag: 'Other', message: 'Content filtered by provider' },
                requestId,
              }
              break
            }

            case 'SafetyStop': {
              executionResult = {
                _tag: 'SafetyStop',
                reason: outcome.reason,
                requestId,
              }
              break
            }

            case 'Interrupted': {
              executionResult = {
                _tag: 'Cancelled',
                reason: { _tag: 'UserInterrupt' },
                requestId,
              }
              break
            }

            case 'ThoughtLimitExceeded': {
              executionResult = {
                _tag: 'Overthinking',
                limit: outcome.limit,
                requestId,
              }
              break
            }
          }

          // ── Circuit breaker ──────────────────────────────────
          const willRetrigger = outcomeWillChainContinue(executionResult)

          if (willRetrigger) {
            const prevCount = trackerState && trackerState.lastResponseText === contentFingerprint
              ? trackerState.consecutiveCount + 1
              : 1

            trackerState = {
              lastResponseText: contentFingerprint,
              consecutiveCount: prevCount,
            }

            if (prevCount >= IDENTICAL_RESPONSE_BREAKER_THRESHOLD) {
              executionResult = {
                _tag: 'SafetyStop',
                reason: {
                  _tag: 'IdenticalResponseCircuitBreaker',
                  threshold: prevCount,
                },
                requestId: executionResult.requestId,
              }
              trackerState = null
            }
          } else {
            trackerState = null
          }

          break
        }
      }
    })

  const getResult = (): ExecuteResult => ({
    result: executionResult,
    commitPolicy,
    usage: turnUsage,
  })

  const getIdenticalResponseTracker = (): IdenticalResponseTracker | null => trackerState

  return { processEvent, getResult, getIdenticalResponseTracker }
}
