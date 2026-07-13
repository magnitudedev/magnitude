/**
 * ATIF Writer Worker
 *
 * Reactively writes the ATIF trajectory to disk after each completed step,
 * so that a partial trajectory is always available even if the process is
 * killed (e.g. Harbor timeout).
 *
 * Uses atomic write (tmp + rename) to guarantee readers never see a
 * truncated JSON file. Gated by AtifAmbient — zero cost when ATIF is
 * disabled or no file path is configured.
 */

import { Effect, SubscriptionRef } from 'effect'
import { Worker, AmbientServiceTag } from '@magnitudedev/event-core'
import { logger } from '@magnitudedev/logger'
import * as fs from 'node:fs/promises'
import * as nodePath from 'node:path'

import type { AppEvent } from '../events'
import { AtifProjection } from '../projections/atif/projection'
import { serializeAtif } from '../projections/atif/serialize'
import { AtifAmbient } from '../ambient/atif-ambient'
import { ConfigAmbient } from '../ambient/config-ambient'
import { SessionOptionsAmbient } from '../ambient/session-ambient'

export const AtifWriter = Worker.define<AppEvent>()({
  name: 'AtifWriter',

  signalHandlers: (on) => [
    on(AtifProjection.signals.stepAdded, (_value, _publish) =>
      Effect.gen(function* () {
        const ambientService = yield* AmbientServiceTag
        const config = ambientService.getValue(AtifAmbient)
        if (!config.enabled || !config.writeFile || !config.filePath) return

        const atifInstance = yield* AtifProjection.Tag
        const atifState = yield* SubscriptionRef.get(atifInstance.state)
        const configState = ambientService.getValue(ConfigAmbient)
        const sessionOptions = ambientService.getValue(SessionOptionsAmbient)
        const trajectory = serializeAtif(atifState.forks, { configState, solo: sessionOptions.solo })

        // Atomic write: tmp file + rename
        const filePath = config.filePath
        const tmpPath = filePath + '.tmp'

        yield* Effect.promise(() =>
          fs.mkdir(nodePath.dirname(filePath), { recursive: true })
        )
        yield* Effect.promise(() =>
          fs.writeFile(tmpPath, JSON.stringify(trajectory, null, 2), 'utf-8')
        )
        yield* Effect.promise(() => fs.rename(tmpPath, filePath))

        logger.debug(
          { filePath, steps: trajectory.steps?.length ?? 0 },
          'ATIF trajectory written (incremental)',
        )
      }),
    ),
  ],
})
