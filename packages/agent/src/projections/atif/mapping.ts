/**
 * AppEvent → ATIF step mapping logic
 */

import type {
  UserMessage,
  TurnStarted,
  TurnOutcomeEvent,
  ThinkingChunk,
  MessageChunkEvent,
  ToolEvent,
  AgentCreated,
  CompactionPrepared,
  Interrupt,
  ObserverOutcome,
  ObservationPart,
  Attachment,
  TurnOutcome,
} from '../../events'
import type {
  AtifStepDraft,
  AtifStepSource,
  AtifMessage,
  AtifToolCall,
  AtifObservationResult,
  AtifMetrics,
  ActiveAtifTurn,
  AtifContentPart,
  AtifImagePart,
  AtifImageSource,
} from './types'
import { JsonValueSchema, type JsonValue } from '@magnitudedev/ai'
import { Schema, Option } from 'effect'

// =============================================================================
// Helpers
// =============================================================================

function timestampToIso(ts: number): string {
  return new Date(ts).toISOString()
}

function assertMediaType(mt: string): AtifImageSource['media_type'] {
  switch (mt) {
    case 'image/jpeg':
    case 'image/png':
    case 'image/gif':
    case 'image/webp':
      return mt
    default:
      return 'image/png'
  }
}

function partsToAtifMessage(parts: readonly ObservationPart[]): AtifMessage {
  const result: AtifContentPart[] = []
  for (const part of parts) {
    if (part.type === 'text') {
      result.push({ type: 'text', text: part.text })
    } else if (part.type === 'image') {
      result.push({
        type: 'image',
        source: { media_type: assertMediaType(part.mediaType), path: 'inline' },
      })
    }
  }
  return result.length === 1 && result[0].type === 'text' ? result[0].text : result
}

function observationToAtifMessage(parts: readonly ObservationPart[] | undefined): AtifMessage {
  if (!parts || parts.length === 0) return ''
  return partsToAtifMessage(parts)
}

/**
 * Convert UserPart[] (content from events.ts) to ATIF message.
 * Handles TextPart, ImagePart, and other part types.
 */
function userPartsToAtifMessage(
  parts: readonly { readonly _tag: string; readonly text?: string; readonly base64?: string; readonly mediaType?: string }[]
): AtifMessage {
  const result: AtifContentPart[] = []
  for (const part of parts) {
    if (part._tag === 'TextPart') {
      result.push({ type: 'text', text: part.text ?? '' })
    } else if (part._tag === 'ImagePart' && part.base64 && part.mediaType) {
      result.push({
        type: 'image',
        source: { media_type: assertMediaType(part.mediaType), path: 'inline' },
      })
    }
  }
  return result.length === 1 && result[0].type === 'text' ? result[0].text : result
}

/**
 * Convert attachments (files) to ATIF content parts appended to the message.
 */
function attachmentsToContentParts(attachments: readonly Attachment[]): AtifImagePart[] {
  const parts: AtifImagePart[] = []
  for (const att of attachments) {
    if (att.type === 'image') {
      parts.push({
        type: 'image',
        source: { media_type: assertMediaType(att.mediaType), path: att.path },
      })
    }
  }
  return parts
}

function buildUserMessage(event: UserMessage): AtifMessage {
  const contentParts = userPartsToAtifMessage(event.content)
  const attachmentParts = attachmentsToContentParts(event.attachments)

  if (typeof contentParts === 'string') {
    if (attachmentParts.length > 0) {
      return [{ type: 'text', text: contentParts }, ...attachmentParts]
    }
    return contentParts
  }

  // contentParts is AtifContentPart[]
  if (attachmentParts.length > 0) {
    return [...contentParts, ...attachmentParts]
  }
  return contentParts.length === 1 && contentParts[0].type === 'text' ? contentParts[0].text : contentParts
}

type JsonRecord = Readonly<Record<string, JsonValue>>
const isJsonValue = Schema.is(JsonValueSchema)

function getExtraFromOutcome(outcome: TurnOutcome): JsonRecord | undefined {
  switch (outcome._tag) {
    case 'Completed':
      return { finishReason: 'stop', toolCallsCount: outcome.completion.toolCallsCount }
    case 'ToolInputValidationFailure':
      return { error: 'tool_input_validation_failure', toolCallId: outcome.toolCallId, toolName: outcome.toolName }
    case 'ToolExecutionError':
      return { error: 'tool_execution_error', toolCallId: outcome.toolCallId, toolName: outcome.toolName, message: outcome.error.message }
    case 'GateRejected':
      return { error: 'gate_rejected', toolCallId: outcome.toolCallId, toolName: outcome.toolName }
    case 'ProviderNotReady':
      return { error: 'provider_not_ready', detail: outcome.detail._tag }
    case 'ConnectionFailure':
      return { error: 'connection_failure', detail: outcome.detail._tag }
    case 'StreamFailed':
      return { error: 'stream_failed', streamFailure: outcome.failure.tag, detail: outcome.failure.detailTag, message: outcome.failure.message }
    case 'ContextWindowExceeded':
      return { error: 'context_window_exceeded' }
    case 'OutputTruncated':
      return { error: 'output_truncated' }
    case 'SafetyStop':
      return { error: 'safety_stop', reason: outcome.reason._tag }
    case 'Cancelled':
      return { error: 'cancelled', reason: outcome.reason._tag }
    case 'Overthinking':
      return { error: 'overthinking', limit: outcome.limit }
    case 'UnexpectedError':
      return { error: 'unexpected_error', message: outcome.detail.message, detail: outcome.detail._tag }
    default:
      return undefined
  }
}

// =============================================================================
// User message → user step
// =============================================================================

export function userMessageToStep(event: UserMessage): AtifStepDraft {
  const message = buildUserMessage(event)
  return {
    timestamp: Option.some(timestampToIso(event.timestamp)),
    source: 'user' as AtifStepSource,
    model_name: Option.none(),
    reasoning_effort: Option.none(),
    message,
    reasoning_content: Option.none(),
    tool_calls: Option.none(),
    observation: Option.none(),
    metrics: Option.none(),
    is_copied_context: Option.none(),
    llm_call_count: Option.none(),
    extra: Option.some({
      messageId: event.messageId,
      forkId: event.forkId,
      ...(event.synthetic ? { autopilot: true } : {}),
      ...(event.taskMode ? { taskMode: true } : {}),
    }),
  }
}

// =============================================================================
// Turn started → initialize partial agent step
// =============================================================================

export function beginActiveTurn(event: TurnStarted, modelId: string | null): ActiveAtifTurn {
  return {
    turnId: event.turnId,
    chainId: event.chainId,
    forkId: event.forkId,
    source: 'agent',
    timestamp: timestampToIso(Date.now()),
    model_name: modelId,
    message: '',
    reasoning_content: '',
    tool_calls: [],
    observation_results: [],
    pendingToolCalls: new Map(),
    metrics: null,
    llm_call_count: 1,
  }
}

// =============================================================================
// Accumulate streaming chunks into partial step
// =============================================================================

export function accumulateThinkingChunk(step: ActiveAtifTurn, event: ThinkingChunk): ActiveAtifTurn {
  return {
    ...step,
    reasoning_content: step.reasoning_content + event.text,
  }
}

export function accumulateMessageChunk(step: ActiveAtifTurn, event: MessageChunkEvent): ActiveAtifTurn {
  return {
    ...step,
    message: step.message + event.text,
  }
}

// =============================================================================
// Tool input ready → add tool call to current step
// =============================================================================

export function addToolCallToStep(step: ActiveAtifTurn, event: ToolEvent): ActiveAtifTurn {
  const lifecycle = event.event as { _tag: string; toolName?: string; toolKey?: string; cached?: boolean }

  const toolCall: AtifToolCall = {
    tool_call_id: event.toolCallId,
    function_name: lifecycle.toolName ?? String(event.toolKey),
    arguments: {}, // populated later from ToolExecutionStarted
    extra: Option.none(),
  }
  return {
    ...step,
    tool_calls: [...step.tool_calls, toolCall],
  }
}

// =============================================================================
// Tool execution ended → add observation to current step
// =============================================================================

export function addObservationToStep(
  step: ActiveAtifTurn,
  event: ToolEvent,
): ActiveAtifTurn {
  const lifecycle = event.event as {
    _tag: string
    result?: unknown
  }
  const rawResult = lifecycle.result

  let content: AtifMessage = ''
  let extra: JsonRecord | undefined

  if (rawResult != null && typeof rawResult === 'object') {
    const result = rawResult as Record<string, unknown>

    // Dispatch on _tag for well-typed handling of all ToolResult variants
    switch (result._tag) {
      case 'Success': {
        const output = result.output
        if (typeof output === 'string') {
          content = output
        } else if (output != null && typeof output === 'object') {
          const out = output as Record<string, unknown>
          if ('parts' in out && Array.isArray(out.parts)) {
            content = observationToAtifMessage(out.parts as readonly ObservationPart[])
          } else {
            try {
              content = JSON.stringify(output)
            } catch {
              content = String(output)
            }
          }
        } else {
          content = String(output ?? '')
        }
        break
      }
      case 'Error': {
        const error = result.error as Record<string, unknown> | undefined
        content = `Error: ${error?.message ?? 'Unknown error'}`
        extra = { error: true }
        break
      }
      case 'Denied': {
        const denial = result.denial
        if (typeof denial === 'string') {
          content = `Denied: ${denial}`
        } else if (isJsonValue(denial)) {
          content = 'Denied'
          extra = { denial }
        } else {
          content = 'Denied'
        }
        break
      }
      case 'Interrupted': {
        content = 'Interrupted'
        extra = { interrupted: true }
        break
      }
      case 'InputRejected': {
        const issue = result.issue as Record<string, unknown> | undefined
        content = `Input rejected: ${issue?.message ?? 'Validation failed'}`
        extra = { inputRejected: true }
        break
      }
      default: {
        // Fallback for unknown result shapes
        if ('parts' in result && Array.isArray(result.parts)) {
          content = observationToAtifMessage(result.parts as readonly ObservationPart[])
        } else if ('output' in result) {
          content = typeof result.output === 'string' ? result.output : JSON.stringify(result.output)
        } else {
          try {
            content = JSON.stringify(rawResult)
          } catch {
            content = String(rawResult)
          }
        }
      }
    }
  } else if (typeof rawResult === 'string') {
    content = rawResult
  }

  const atifResult: AtifObservationResult = {
    source_call_id: Option.some(event.toolCallId),
    content: Option.some(content),
    subagent_trajectory_ref: Option.none(),
    ...(extra ? { extra: Option.some(extra) } : { extra: Option.none() }),
  }

  return {
    ...step,
    observation_results: [...step.observation_results, atifResult],
  }
}

// =============================================================================
// Turn outcome → finalize step with metrics
// =============================================================================

/**
 * Whether a turn outcome indicates the LLM call failed without producing
 * any output (e.g., connection failure, unexpected transport error).
 * These steps record that an LLM call was attempted but no inference
 * completed, so llm_call_count is set to 0 and metrics/reasoning are
 * omitted since no tokens were processed.
 *
 * NOTE: ATIF has no standard mechanism for recording LLM call errors.
 * Using llm_call_count=0 is a pragmatic choice — it correctly signals
 * that no LLM inference occurred on this step, and allows consumers to
 * distinguish these from real behavioral turns (llm_call_count > 0).
 * Error details are carried in extra.error.
 */
function isFailedLlmCall(outcome: TurnOutcome): boolean {
  return outcome._tag === 'ConnectionFailure'
    || outcome._tag === 'StreamFailed'
    || outcome._tag === 'UnexpectedError'
}

export function finalizeAgentStep(
  partial: ActiveAtifTurn,
  event: TurnOutcomeEvent
): AtifStepDraft {
  const outcomeExtra = getExtraFromOutcome(event.outcome)

  // Failed LLM calls (connection failure, transport error) with no output
  // get llm_call_count=0 and omit metrics/reasoning since no inference completed.
  const isNoLlm = isFailedLlmCall(event.outcome) &&
    partial.message.trim().length === 0 &&
    partial.tool_calls.length === 0 &&
    partial.observation_results.length === 0

  const llmCallCount = isNoLlm ? 0 : partial.llm_call_count

  // Use modelId from turn_outcome if model_name wasn't set at turn_started
  const modelName = partial.model_name ?? event.modelId ?? null

  // Only compute metrics when an LLM inference actually occurred
  let finalMetrics: AtifMetrics | undefined
  if (!isNoLlm) {
    const metrics: AtifMetrics | undefined =
      event.inputTokens != null || event.outputTokens != null
        ? {
            prompt_tokens: event.inputTokens != null ? Option.some(event.inputTokens) : Option.none(),
            completion_tokens: event.outputTokens != null ? Option.some(event.outputTokens) : Option.none(),
            cached_tokens: event.cacheReadTokens != null ? Option.some(event.cacheReadTokens) : Option.none(),
            cost_usd: Option.none(),
            prompt_token_ids: Option.none(),
            completion_token_ids: Option.none(),
            logprobs: Option.none(),
            extra: Option.none(),
          }
        : undefined

    const providerMetrics: Record<string, JsonValue> = {}
    if (event.cacheWriteTokens != null) {
      providerMetrics.cache_creation_input_tokens = event.cacheWriteTokens
    }
    if (event.providerId) {
      providerMetrics.provider_id = event.providerId
    }
    if (event.modelId) {
      providerMetrics.model_id = event.modelId
    }

    finalMetrics =
      metrics || Object.keys(providerMetrics).length > 0 || event.cost != null
        ? {
            prompt_tokens: metrics?.prompt_tokens ?? Option.none(),
            completion_tokens: metrics?.completion_tokens ?? Option.none(),
            cached_tokens: metrics?.cached_tokens ?? Option.none(),
            cost_usd: event.cost != null ? Option.some(event.cost) : Option.none(),
            prompt_token_ids: Option.none(),
            completion_token_ids: Option.none(),
            logprobs: Option.none(),
            extra: Object.keys(providerMetrics).length > 0 ? Option.some(providerMetrics) : Option.none(),
          }
        : undefined
  }

  const step: AtifStepDraft = {
    timestamp: Option.some(partial.timestamp),
    source: 'agent',
    model_name: modelName ? Option.some(modelName) : Option.none(),
    reasoning_effort: Option.none(),
    message: partial.message.trim() || '',
    // reasoning_content omitted when no LLM inference completed
    ...(!isNoLlm && partial.reasoning_content.trim()
      ? { reasoning_content: Option.some(partial.reasoning_content.trim()) }
      : { reasoning_content: Option.none() }),
    ...(partial.tool_calls.length > 0
      ? { tool_calls: Option.some(partial.tool_calls) }
      : { tool_calls: Option.none() }),
    ...(partial.observation_results.length > 0
      ? {
          observation: Option.some({
            results: partial.observation_results,
          }),
        }
      : { observation: Option.none() }),
    ...(finalMetrics ? { metrics: Option.some(finalMetrics) } : { metrics: Option.none() }),
    is_copied_context: Option.none(),
    llm_call_count: Option.some(llmCallCount),
    extra: Option.some({
      turnId: event.turnId,
      chainId: event.chainId,
      forkId: event.forkId,
      outcome: event.outcome._tag,
      requestId: event.outcome.requestId,
      ...(event.providerId != null ? { providerId: event.providerId } : {}),
      ...(event.modelId != null ? { modelId: event.modelId } : {}),
      ...(outcomeExtra ?? {}),
    }),
  }

  return step
}

// =============================================================================
// Agent created → agent step with spawnWorker
// =============================================================================

export function agentCreatedToStep(event: AgentCreated, toolCallId: string): AtifStepDraft {
  return {
    timestamp: Option.some(timestampToIso(Date.now())),
    source: 'agent',
    model_name: Option.none(),
    reasoning_effort: Option.none(),
    message: '',
    reasoning_content: Option.none(),
    tool_calls: Option.some([
      {
        tool_call_id: toolCallId,
        function_name: 'spawnWorker',
        arguments: {
          role: event.role,
          taskId: event.taskId,
          mode: event.mode,
          ...(event.message ? { message: event.message } : {}),
        },
        extra: Option.none(),
      },
    ]),
    observation: Option.some({
      results: [
        {
          source_call_id: Option.some(toolCallId),
          content: Option.none(),
          subagent_trajectory_ref: Option.some([
            {
              trajectory_id: Option.some(event.agentId),
              trajectory_path: Option.none(),
              session_id: Option.none(),
              extra: Option.none(),
            },
          ]),
          extra: Option.none(),
        },
      ],
    }),
    metrics: Option.none(),
    is_copied_context: Option.none(),
    extra: Option.some({
      agentId: event.agentId,
      forkId: event.forkId,
      parentForkId: event.parentForkId,
      taskId: event.taskId,
    }),
    llm_call_count: Option.some(0),
  }
}

// =============================================================================
// Interrupt → user step (user intervention)
// =============================================================================

export function interruptToStep(event: Interrupt): AtifStepDraft {
  return {
    timestamp: Option.some(timestampToIso(Date.now())),
    source: 'user',
    model_name: Option.none(),
    reasoning_effort: Option.none(),
    message: event.forkId === null ? 'Agent interrupted' : 'Worker interrupted',
    reasoning_content: Option.none(),
    tool_calls: Option.none(),
    observation: Option.none(),
    metrics: Option.none(),
    is_copied_context: Option.none(),
    extra: Option.some({
      forkId: event.forkId,
    }),
    llm_call_count: Option.some(0),
  }
}

// =============================================================================
// Compaction prepared → system step (ATIF context_management boundary)
// =============================================================================

export function compactionPreparedToStep(event: CompactionPrepared): AtifStepDraft {
  const observationResults: AtifObservationResult[] = []

  // If not a fallback, extract the compaction summary from compactResult
  if (!event.isFallback && event.compactResult) {
    const summary = event.compactResult.summary
    if (summary) {
      observationResults.push({
        source_call_id: Option.some(`compaction-${event.turn.turnId}`),
        content: Option.some(summary),
        subagent_trajectory_ref: Option.none(),
        extra: Option.none(),
      })
    }
  }

  return {
    timestamp: Option.some(timestampToIso(Date.now())),
    source: 'system',
    model_name: Option.none(),
    reasoning_effort: Option.none(),
    message: 'Context compaction performed',
    reasoning_content: Option.none(),
    tool_calls: Option.none(),
    ...(observationResults.length > 0
      ? { observation: Option.some({ results: observationResults }) }
      : { observation: Option.none() }),
    metrics: Option.none(),
    is_copied_context: Option.none(),
    extra: Option.some({
      forkId: event.forkId,
      turnId: event.turn.turnId,
      context_management: {
        type: 'compaction',
        boundary: 'replace',
        compactedMessageCount: event.compactedMessageCount,
        ...(event.isFallback ? { isFallback: true } : {}),
        ...(event.inputTokens != null ? { inputTokens: event.inputTokens } : {}),
        ...(event.outputTokens != null ? { outputTokens: event.outputTokens } : {}),
      },
    }),
    llm_call_count: Option.some(0),
  }
}

// =============================================================================
// Agent killed → terminal agent step
// =============================================================================

export function agentKilledToStep(agentId: string, reason: string): AtifStepDraft {
  return {
    timestamp: Option.some(timestampToIso(Date.now())),
    source: 'agent',
    model_name: Option.none(),
    reasoning_effort: Option.none(),
    message: `Agent killed: ${reason}`,
    reasoning_content: Option.none(),
    tool_calls: Option.none(),
    observation: Option.none(),
    metrics: Option.none(),
    is_copied_context: Option.none(),
    extra: Option.some({
      agentKilled: true,
      agentId,
      reason,
    }),
    llm_call_count: Option.some(0),
  }
}

// =============================================================================
// Observer outcome → system step
// =============================================================================

function observerOutcomeMessage(event: ObserverOutcome): string {
  return JSON.stringify(
    event.escalate
      ? { escalate: true, justification: event.justification }
      : { escalate: false },
  )
}

export function observerOutcomeToStep(event: ObserverOutcome): AtifStepDraft {
  return {
    timestamp: Option.some(timestampToIso(Date.now())),
    source: 'system',
    model_name: Option.none(),
    reasoning_effort: Option.none(),
    message: observerOutcomeMessage(event),
    reasoning_content: Option.some(event.reasoning),
    tool_calls: Option.none(),
    observation: Option.none(),
    metrics: Option.none(),
    is_copied_context: Option.none(),
    extra: Option.some({
      observer: true,
      observedTurnId: event.observedTurnId,
      observerTurnId: event.observerTurnId,
      escalate: event.escalate,
      justification: event.justification,
      chainId: event.chainId,
    }),
    llm_call_count: Option.some(0),
  }
}
