import { Effect } from 'effect'
import { Worker } from '@magnitudedev/event-core'
import type { AppEvent } from '../events'
import { ChatTitleProjection } from '../projections/chat-title'
import { logger } from '@magnitudedev/logger'
import { updateTraceMeta } from '@magnitudedev/tracing'
import { ChatPersistence } from '../persistence/chat-persistence-service'

export const ChatTitleWorker = Worker.define<AppEvent>()({
  name: 'ChatTitleWorker',

  signalHandlers: (on) => [
    on(
      ChatTitleProjection.signals.chatTitleResolved,
      (value) =>
        Effect.gen(function* () {
          const persistence = yield* ChatPersistence
          yield* persistence.saveSessionMetadata({ chatName: value.title }).pipe(
            Effect.catchAll((error) =>
              Effect.sync(() => logger.error(
                { error, title: value.title },
                '[chat-title-worker] Failed to persist chat title metadata',
              )),
            ),
          )
          updateTraceMeta({ chatName: value.title })
        }),
    ),
  ],
})
