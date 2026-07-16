/**
 * Chat Title Service — AI-generated session titles from first user message.
 *
 * Architecture:
 * - Layer.scoped service with Context.Tag
 * - Exposes a `generate` method for explicit title generation
 * - Called by the ChatTitleWorker (thin signal listener) on first user message
 *
 * Fire-and-forget: the title is generated asynchronously and updates
 * session metadata when ready.
 */

import { Cause, Context, Duration, Effect, Layer, Stream } from 'effect'
import * as HttpClient from '@effect/platform/HttpClient'
import { formatStreamFailureMessage, Prompt } from '@magnitudedev/ai'
import type { ModelStreamTerminal, TextPart, UserPart } from '@magnitudedev/ai'
import { logger } from '@magnitudedev/logger'
import { AmbientServiceTag, type AmbientService } from '@magnitudedev/event-core'
import { AgentModelResolver } from '../model/model-resolver'
import { connectionRetrySchedule } from '../util/retry-backoff'
import { modelAttemptRetryability } from '../errors'
import { CHAT_TITLE_PROMPT } from '../util/title-prompts'

export const CHAT_TITLE_MAX_TOKENS = 100

// =============================================================================
// Types
// =============================================================================

export interface ChatTitleService {
  /**
   * Generate a title from the user's first message.
   * Returns the generated title string, or null on failure/timeout.
   * Pure library — no side effects (no persistence, no trace updates).
   */
  readonly generate: (userMessage: string) => Effect.Effect<string | null, never, AmbientService>
}

export class ChatTitleServiceTag extends Context.Tag('ChatTitleService')<
  ChatTitleServiceTag,
  ChatTitleService
>() {}

// =============================================================================
// Helpers
// =============================================================================

function extractTextFromParts(parts: readonly UserPart[]): string {
  return parts
    .filter((p): p is TextPart => p._tag === 'TextPart')
    .map((p) => p.text)
    .join(' ')
}

function streamTerminalErrorMessage(terminal: ModelStreamTerminal): string | null {
  switch (terminal._tag) {
    case 'StreamCompleted':
      return null
    case 'StreamFailed':
      return formatStreamFailureMessage(terminal.cause)
  }
}

// Re-export for worker convenience
export { extractTextFromParts }

// =============================================================================
// Live Layer
// =============================================================================

export const ChatTitleServiceLive = Layer.scoped(
  ChatTitleServiceTag,
  Effect.gen(function* () {
    const modelResolver = yield* AgentModelResolver
    const httpClient = yield* HttpClient.HttpClient

    const generate = (userMessage: string): Effect.Effect<string | null, never, AmbientService> =>
      Effect.gen(function* () {
        logger.info('[chat-title-service] generate() called')

        logger.info('[chat-title-service] Resolving title model from the primary slot...')
        const titleModel = yield* modelResolver.resolvePrimary('leader')
        logger.info({ modelId: titleModel.modelId, modelSource: titleModel.modelSource }, '[chat-title-service] Title model resolved')

        const prompt = Prompt.from({
          messages: [
            {
              _tag: 'UserMessage' as const,
              parts: [
                {
                  _tag: 'TextPart' as const,
                  text: `${CHAT_TITLE_PROMPT}\n\nUser message: "${userMessage.slice(0, 500)}"`,
                },
              ],
            },
          ],
        })

        logger.info('[chat-title-service] Starting model stream call...')
        const streamResult = yield* Effect.gen(function* () {
          const result = yield* titleModel.model.stream(prompt, [], { maxTokens: CHAT_TITLE_MAX_TOKENS }).pipe(
            Effect.tap(() => Effect.sync(() => logger.info('[chat-title-service] Stream started'))),
            Effect.tapError((e) => Effect.sync(() => logger.error({ error: String(e) }, '[chat-title-service] Stream error before timeout'))),
          )
          logger.info('[chat-title-service] Stream call returned successfully')
          return result
        }).pipe(
          Effect.timeoutTo({
            duration: Duration.seconds(10),
            onSuccess: (result) => result,
            onTimeout: () => {
              logger.info('[chat-title-service] Stream call timed out after 10s')
              return null
            },
          }),
          Effect.retry({
            schedule: connectionRetrySchedule,
            while: (err) => modelAttemptRetryability(err)._tag === 'UpstreamRetryable',
          }),
          Effect.catchAll((e) => Effect.sync(() => {
            logger.error({ error: String(e) }, '[chat-title-service] Caught error in stream pipeline')
            return null
          })),
          Effect.provideService(HttpClient.HttpClient, httpClient),
        )

        if (!streamResult) {
          logger.info('[chat-title-service] Title generation timed out or failed, keeping fallback title')
          return null
        }

        const collected = yield* Stream.runFold(
          streamResult.events,
          { text: '', streamError: null as string | null },
          (state, event) => {
            if (event._tag === 'message_delta') return { ...state, text: state.text + event.text }
            if (event._tag === 'stream_end') {
              return { ...state, streamError: streamTerminalErrorMessage(event.terminal) }
            }
            return state
          },
        )

        const title = collected.text.trim().slice(0, 50)
        if (title) {
          if (collected.streamError) {
            logger.info(
              { title, streamError: collected.streamError },
              '[chat-title-service] Using partial title from failed generation stream',
            )
          } else {
            logger.info({ title }, '[chat-title-service] Generated chat title')
          }
          return title
        }

        if (collected.streamError) {
          logger.info({ streamError: collected.streamError }, '[chat-title-service] Title generation stream failed without usable text, keeping fallback title')
        } else {
          logger.info('[chat-title-service] Title generation returned empty, keeping fallback title')
        }
        return null
      }).pipe(
        Effect.catchAllCause((cause) => {
          if (Cause.isInterruptedOnly(cause)) return Effect.succeed(null)
          logger.error({ cause: Cause.pretty(cause) }, '[chat-title-service] Failed to generate title')
          return Effect.succeed(null)
        }),
      )

    return { generate }
  }),
)
