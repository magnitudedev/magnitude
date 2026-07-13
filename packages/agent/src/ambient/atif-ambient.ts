import { Ambient, AmbientServiceTag } from '@magnitudedev/event-core'
import { Effect } from 'effect'

/**
 * Ambient for ATIF (Agent Trajectory Interchange Format) trajectory generation.
 *
 * Standalone ambient — ATIF config does not belong on ConfigAmbient
 * since it's an opt-in feature with no relationship to model/role config.
 *
 * Disabled by default. Zero cost when disabled — the projection checks
 * this ambient first and short-circuits if not enabled.
 */

export interface AtifConfig {
  readonly enabled: boolean
  readonly writeFile: boolean
  readonly filePath: string | null
  readonly streamSteps: boolean
  readonly stepsPath: string | null
}

export const DEFAULT_ATIF_CONFIG: AtifConfig = {
  enabled: false,
  writeFile: false,
  filePath: null,
  streamSteps: false,
  stepsPath: null,
}

export const AtifAmbient = Ambient.define<AtifConfig, never>({
  name: 'Atif',
  initial: Effect.succeed(DEFAULT_ATIF_CONFIG),
})

export function publishAtifConfig(config: AtifConfig) {
  return Effect.gen(function* () {
    const ambientService = yield* AmbientServiceTag
    yield* ambientService.update(AtifAmbient, config)
  })
}
