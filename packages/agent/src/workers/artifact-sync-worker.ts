/**
 * ArtifactSyncWorker
 *
 * Writes artifact changes through to disk when an artifact has a bound syncPath.
 */

import { Effect } from 'effect'
import { Worker } from '@magnitudedev/event-core'
import { logger } from '@magnitudedev/logger'
import type { AppEvent } from '../events'
import { ArtifactProjection } from '../projections/artifact'

export const ArtifactSyncWorker = Worker.define<AppEvent>()({
  name: 'ArtifactSyncWorker',

  signalHandlers: (on) => {
    const writeArtifactToDisk = (id: string, read: Parameters<Parameters<typeof on>[1]>[2]) =>
      Effect.gen(function* () {
        const state = yield* read(ArtifactProjection)
        const artifact = state.artifacts.get(id)
        if (!artifact?.syncPath) return

        yield* Effect.tryPromise({
          try: () => Bun.write(artifact.syncPath!, artifact.content),
          catch: (error) => error instanceof Error ? error : new Error(String(error)),
        }).pipe(
          Effect.catchAll((error) =>
            Effect.sync(() =>
              logger.error(
                { artifactId: id, path: artifact.syncPath, error: error.message },
                '[ArtifactSyncWorker] Failed to write synced artifact to disk'
              )
            )
          )
        )
      })

    return [
      on(ArtifactProjection.signals.artifactChanged, ({ id }, _publish, read) =>
        writeArtifactToDisk(id, read)
      ),
      on(ArtifactProjection.signals.artifactSynced, ({ id }, _publish, read) =>
        writeArtifactToDisk(id, read)
      ),
    ]
  },
})