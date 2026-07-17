import { Effect } from 'effect'
import { Worker } from '@magnitudedev/event-core'
import type { AppEvent } from '../events'
import { ChatTitleServiceTag } from './chat-title-service'
import { UserMessageResolutionProjection } from '../projections/user-message-resolution'
import { ChatTitleProjection } from '../projections/chat-title'
import { logger } from '@magnitudedev/logger'
import { updateTraceMeta } from '@magnitudedev/tracing'
import { ChatPersistence } from '../persistence/chat-persistence-service'
import { DEFAULT_CHAT_NAME } from '../constants'
import { AgentModelOperationContextTag } from '../model/agent-model'
import { fallbackChatTitle } from '../util/title-prompts'

export const ChatTitleWorker = Worker.define<AppEvent>()({
  name: 'ChatTitleWorker',

  signalHandlers: (on) => [
    on(
      UserMessageResolutionProjection.signals.userMessageResolved,
      (value, publish, read) =>
        Effect.gen(function* () {
          logger.info({ forkId: value.forkId, synthetic: value.synthetic }, '[chat-title-worker] Signal received')

          // Only root fork, only real user messages
          if (value.forkId !== null || value.synthetic) {
            logger.info('[chat-title-worker] Skipping: forked or synthetic')
            return
          }

          // Dedup: check ChatTitleProjection state — if chatName is already
          // set (non-default), skip. This handles duplicate signals within a session.
          const metadata = yield* read(ChatTitleProjection)
          if (metadata.chatName !== null && metadata.chatName !== DEFAULT_CHAT_NAME) {
            logger.info({ chatName: metadata.chatName }, '[chat-title-worker] Skipping: title already set (projection)')
            return
          }

          // Also check persistence — on session resume the projection starts fresh
          // but the persisted title may already be set (including manual renames).
          const persistence = yield* ChatPersistence
          const existingMeta = yield* persistence.getSessionMetadata().pipe(
            Effect.catchAll((error) =>
              Effect.sync(() => {
                logger.error({ error }, '[chat-title-worker] Failed to read persisted metadata')
                return null
              }),
            ),
          )
          if (existingMeta === null) return
          if (existingMeta.chatName !== DEFAULT_CHAT_NAME) {
            logger.info({ chatName: existingMeta.chatName }, '[chat-title-worker] Skipping: persisted title already set')
            return
          }

          // Extract text content
          const text = value.text
          if (!text.trim()) {
            logger.info('[chat-title-worker] Skipping: no text content')
            return
          }

          // Establish a useful title before starting the model call. Besides
          // making title generation resilient to local/offline model failures,
          // this persisted event is the one-attempt guard: later user messages
          // see a non-default title and never start another title request.
          const fallbackTitle = fallbackChatTitle(text)
          if (fallbackTitle === null) return

          yield* persistence.saveSessionMetadata({ chatName: fallbackTitle }).pipe(
            Effect.catchAll((error) =>
              Effect.sync(() => logger.error(
                { error, fallbackTitle },
                '[chat-title-worker] Failed to persist fallback title metadata',
              )),
            ),
          )
          updateTraceMeta({ chatName: fallbackTitle })
          yield* publish({
            type: 'chat_title_generated',
            forkId: null,
            title: fallbackTitle,
            timestamp: Date.now(),
          })

          logger.info({ textPreview: text.slice(0, 100) }, '[chat-title-worker] Delegating to ChatTitleService')

          // Fire-and-forget: run AI generation in background fiber so signal handler returns immediately
          yield* Effect.gen(function* () {
            const service = yield* ChatTitleServiceTag
            const title = yield* service.generate(text).pipe(
              Effect.provideService(AgentModelOperationContextTag, {
                operationKind: 'title',
                operationId: value.messageId,
                relatedMessageId: value.messageId,
                forkId: value.forkId,
              }),
            )

            if (title === null) {
              logger.info({ fallbackTitle }, '[chat-title-worker] Title generation returned null, keeping fallback title')
              return
            }

            // Never replace the fallback with the sentinel default, which
            // would re-enable title generation on the next user message.
            if (title === DEFAULT_CHAT_NAME) {
              logger.info('[chat-title-worker] Title generation returned the default title, keeping fallback title')
              return
            }

            logger.info({ title }, '[chat-title-worker] Title generated, saving metadata and publishing event')

            // Save to persistence (reuse outer `persistence` reference)
            yield* persistence.saveSessionMetadata({ chatName: title })

            // Update trace metadata
            updateTraceMeta({ chatName: title })

            // Publish event so ChatTitleProjection can emit a signal
            yield* publish({ type: 'chat_title_generated', forkId: null, title, timestamp: Date.now() })
          }).pipe(
            Effect.catchAll((error) =>
              Effect.sync(() => logger.error({ error }, '[chat-title-worker] Title generation side effect failed')),
            ),
            Effect.forkScoped,
          )
        }),
    ),
  ],
})
