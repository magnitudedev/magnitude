/**
 * Chat Title Worker
 *
 * Automatically generates chat titles from conversation context.
 * Triggers on user_message events, generates a title using the secondary model,
 * persists it, and publishes a chat_title_generated event.
 */

import { Effect } from 'effect'
import { Worker } from '@magnitudedev/event-core'
import { logger } from '@magnitudedev/logger'
import type { AppEvent } from '../events'
import { MemoryProjection, type ForkMemoryState } from '../projections/memory'
import { UserMessageResolutionProjection } from '../projections/user-message-resolution'
import { ChatPersistence } from '../persistence/chat-persistence-service'
import { ModelResolver, GenerateChatTitle } from '@magnitudedev/providers'
import { updateTraceMeta } from '@magnitudedev/tracing'
import { withTraceScope } from '../tracing'
import { DEFAULT_CHAT_NAME } from '../constants'
import { textOf } from '../content'

const MAX_USER_MESSAGES_TO_TRY = 5

export const ChatTitleWorker = Worker.define<AppEvent>()({
  name: 'ChatTitleWorker',

  eventHandlers: {},

  signalHandlers: on => [
    on(UserMessageResolutionProjection.signals.userMessageResolved, (value, publish, read) => Effect.gen(function* () {
      // Only handle root fork messages
      if (value.forkId !== null) return

      // Read memory to get conversation and count user messages
      const memoryState: ForkMemoryState = yield* read(MemoryProjection)

      // Count user messages and build conversation parts
      const parts: string[] = []
      let userMessageCount = 0

      for (const msg of memoryState.messages) {
        if (msg.type === 'inbox') {
          for (const entry of msg.timeline) {
            if (entry.kind === 'user_message') {
              parts.push(entry.text)
              userMessageCount++
            } else if (entry.kind === 'agent_block') {
              for (const atom of entry.atoms) {
                if (atom.kind === 'message' && atom.direction === 'to_lead') {
                  parts.push(`<assistant>\n${atom.text}\n</assistant>`)
                }
              }
            }
          }
        } else if (msg.type === 'assistant_turn') {
          parts.push(`<assistant>\n${textOf(msg.content)}\n</assistant>`)
        }
      }

      // Give up after too many attempts
      if (userMessageCount > MAX_USER_MESSAGES_TO_TRY) return

      if (parts.length === 0) {
        logger.warn('[ChatTitleWorker] No messages found for title generation')
        return
      }

      // Check if title already exists
      const persistence = yield* ChatPersistence
      const metadata = yield* persistence.getSessionMetadata().pipe(
        Effect.catchAll(() => Effect.succeed({ chatName: '' }))
      )

      if (metadata.chatName && metadata.chatName !== DEFAULT_CHAT_NAME) {
        return
      }

      const conversation = parts.join('\n\n')

      logger.info({ messageCount: parts.length, userMessageCount }, '[ChatTitleWorker] Generating chat title')

      const runtime = yield* ModelResolver
      const model = yield* runtime.resolve('explorer')
      const result = yield* withTraceScope(
        { metadata: { callType: 'title', forkId: value.forkId ?? null } },
        model.invoke(
          GenerateChatTitle,
          { conversation, defaultName: DEFAULT_CHAT_NAME },
        ),
      ).pipe(
        Effect.catchAll((error) => {
          logger.error({ error: String(error) }, '[ChatTitleWorker] Failed to generate chat title')
          return Effect.succeed(null)
        })
      )

      if (!result) return

      const title = result.title

      // If title is still generic, don't save — wait for more context
      if (title === DEFAULT_CHAT_NAME) {
        logger.info({ title, userMessageCount }, '[ChatTitleWorker] Title generation returned generic title, will retry')
        return
      }

      logger.info({ title }, '[ChatTitleWorker] Chat title generated successfully')

      // Persist title
      yield* persistence.saveSessionMetadata({ chatName: title }).pipe(
        Effect.catchAll((error) => {
          logger.error({ error: String(error) }, '[ChatTitleWorker] Failed to save chat title')
          return Effect.void
        })
      )

      logger.info({ title }, '[ChatTitleWorker] Chat title saved')

      // Update trace metadata with the title
      try { updateTraceMeta({ title }) } catch {}

      // Publish event so ChatTitleProjection can emit signal to CLI
      yield* publish({
        type: 'chat_title_generated',
        forkId: null,
        title,
      })
    }).pipe(
      Effect.catchAllCause(cause =>
        Effect.sync(() => {
          logger.error({ cause: cause.toString() }, '[ChatTitleWorker] Unexpected error in chat title generation')
        })
      )
    )),
  ],
})
