import { Effect } from 'effect'
import { Worker } from '@magnitudedev/event-core'
import { logger } from '@magnitudedev/logger'
import type { AppEvent } from '../events'
import { ChatPersistence } from '../persistence/chat-persistence-service'
import { TaskGraphProjection, getSessionTitleFromTaskGraph } from '../projections/task-graph'

export const SessionTitleWorker = Worker.define<AppEvent>()({
  name: 'SessionTitleWorker',

  signalHandlers: (on) => [
    on(TaskGraphProjection.signals.taskCreated, (_value, _publish, read) => Effect.gen(function* () {
      const taskGraph = yield* read(TaskGraphProjection)
      const title = getSessionTitleFromTaskGraph(taskGraph)
      if (title === null) return

      const persistence = yield* ChatPersistence
      const metadata = yield* persistence.getSessionMetadata()
      if (metadata.chatName === title) return

      yield* persistence.saveSessionMetadata({ chatName: title })
    }).pipe(
      Effect.catchAll((error) => Effect.sync(() => {
        logger.error({ error }, '[SessionTitleWorker] Failed to persist session title')
      }))
    )),
  ],
})
