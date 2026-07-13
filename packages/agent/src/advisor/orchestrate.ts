/**
 * Shared advisor orchestration utilities.
 *
 * Core advisor execution logic factored out for use by both the
 * `message_advisor` tool and the automatic escalation observer path.
 */

import { Effect, Schema, Stream } from 'effect'
import * as HttpClient from '@effect/platform/HttpClient'
import { formatStreamFailureMessage, type FinishReason, type ModelStreamResult, type ModelStreamTerminal, type StreamFailure } from '@magnitudedev/ai'
import { advisorPrompt } from '@magnitudedev/roles'
import { AgentModelResolver } from '../model/model-resolver'
import { getAgentByForkId, type AgentLifecycleState } from '../projections/agent-lifecycle'
import { formatModelAttemptFailure, modelAttemptRetryability, presentModelAttemptFailure, type AgentModelAttemptFailure } from '../errors'
import { connectionRetrySchedule } from '../util/retry-backoff'
import { WindowStateReaderTag } from '../tools/window-reader'
import { AgentStateReaderTag } from '../tools/fork'
import { advisorWindowToPrompt } from '../window/render'
import { ToolErrorSchema } from '../tools/errors'

import { Fork } from '@magnitudedev/event-core'

const { ForkContext } = Fork

const ADVISOR_MAX_TOKENS = 1200

const MARKER_AUTOPILOT_OFF = '[AUTOPILOT_OFF]'

export function buildAdvisorSystemPrompt(): string {
  return advisorPrompt.raw
}

export function parseAutopilotResponse(text: string):
  | { _tag: 'message'; content: string }
  | { _tag: 'finish'; content: string | null }
{
  const trimmed = text.trim()
  if (trimmed.endsWith(MARKER_AUTOPILOT_OFF)) {
    const before = trimmed.slice(0, trimmed.length - MARKER_AUTOPILOT_OFF.length).trim()
    return before.length > 0
      ? { _tag: 'finish', content: before }
      : { _tag: 'finish', content: null }
  }
  return { _tag: 'message', content: trimmed }
}

export const AdvisorErrorSchema = ToolErrorSchema('AdvisorError', {})
export type AdvisorError = Schema.Schema.Type<typeof AdvisorErrorSchema>

export function advisorError(message: string): AdvisorError {
  return { _tag: 'AdvisorError', message }
}

export function streamStartFailureMessage(err: AgentModelAttemptFailure): string {
  const presented = presentModelAttemptFailure(err)
  return presented.llmFeedback || presented.message || formatModelAttemptFailure(err)
}

export function streamErrorMessage(failure: StreamFailure): string {
  return formatStreamFailureMessage(failure)
}

function streamTerminalErrorMessage(terminal: ModelStreamTerminal): string | null {
  switch (terminal._tag) {
    case 'StreamCompleted':
      return null
    case 'StreamFailed':
      return streamErrorMessage(terminal.cause)
  }
}

export function callerLabel(args: {
  readonly forkId: string | null
  readonly roleId: string
  readonly agentState: AgentLifecycleState
}): string {
  if (args.forkId === null) return 'coordinator'
  const agent = getAgentByForkId(args.agentState, args.forkId)
  if (!agent) return args.roleId
  return `${agent.agentId} (${agent.role})`
}

export function collectAdvisorText(
  streamResult: ModelStreamResult,
): Effect.Effect<{
  readonly text: string
  readonly finishReason: FinishReason | null
}, AdvisorError> {
  const folded = Stream.runFold(
    streamResult.events,
    { text: '', finishReason: null as FinishReason | null, streamError: null as string | null },
    (state, event) => {
      switch (event._tag) {
        case 'message_delta':
          return { ...state, text: state.text + event.text }
        case 'stream_end': {
          switch (event.terminal._tag) {
            case 'StreamCompleted':
              return { ...state, finishReason: event.terminal.finishReason }
            case 'StreamFailed':
              return { ...state, streamError: streamTerminalErrorMessage(event.terminal) }
          }
        }
        default:
          return state
      }
    },
  )

  return folded.pipe(
    Effect.flatMap((result): Effect.Effect<{
      readonly text: string
      readonly finishReason: FinishReason | null
    }, AdvisorError> => {
      if (result.streamError) {
        return Effect.fail(advisorError(`Advisor stream error: ${result.streamError}`))
      }

      const text = result.text.trim()
      if (!text) {
        return Effect.fail(advisorError('Advisor returned an empty response.'))
      }

      if (result.finishReason === 'content_filter') {
        return Effect.fail(advisorError('Advisor response was blocked by content filtering.'))
      }

      if (result.finishReason === 'length') {
        return Effect.succeed({
          text: `${text}\n\n[Advisor response truncated by output limit.]`,
          finishReason: result.finishReason,
        })
      }

      return Effect.succeed({ text, finishReason: result.finishReason })
    }),
  )
}

/**
 * Core execution for the manual message_advisor tool.
 */
export function executeMessageAdvisor(input: { readonly message: string }) {
  return Effect.gen(function* () {
    const message = input.message.trim()
    if (!message) {
      return yield* Effect.fail(advisorError('message_advisor requires a non-empty message.'))
    }

    const { forkId, roleId } = yield* ForkContext
    const windowReader = yield* WindowStateReaderTag
    const windowState = yield* windowReader.getWindowState(forkId)
    if (!windowState) {
      return yield* Effect.fail(advisorError('Advisor could not read the current context window.'))
    }

    const prompt = advisorWindowToPrompt({
      windowState,
      systemPrompt: buildAdvisorSystemPrompt(),
      autopilotEnabled: false,
      advisorLastAutopilotKnowledge: null,
      messageAdvisorText: message,
    })

    const modelResolver = yield* AgentModelResolver
    const httpClient = yield* HttpClient.HttpClient
    const advisorModel = yield* modelResolver.resolvePrimary('advisor', 'advisor')
    const maxTokens = Math.min(advisorModel.profile.maxOutputTokens, ADVISOR_MAX_TOKENS)

    const streamResult = yield* advisorModel.model.stream(prompt, [], { maxTokens }).pipe(
      Effect.provideService(HttpClient.HttpClient, httpClient),
      Effect.retry({
        schedule: connectionRetrySchedule,
        while: (err) => modelAttemptRetryability(err)._tag === 'UpstreamRetryable',
      }),
      Effect.mapError((err) => advisorError(`Advisor call failed: ${streamStartFailureMessage(err)}`)),
    )

    const result = yield* collectAdvisorText(streamResult)
    return result.text
  })
}


