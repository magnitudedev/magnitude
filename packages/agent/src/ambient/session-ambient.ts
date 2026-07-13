import { Ambient, AmbientServiceTag } from '@magnitudedev/event-core'
import { Effect } from 'effect'

/**
 * Ambient for session-level options (CLI flags, runtime overrides).
 *
 * Set once at session start, never changes during the session.
 * Accessible everywhere via AmbientServiceTag — no Layer wiring needed.
 */

export interface SessionOptions {
  readonly sessionId: string
  readonly disableShellSafeguards: boolean
  readonly disableCwdSafeguards: boolean
  readonly timezone: string | null
  readonly vcsAvailable: boolean
  readonly headless: boolean
  readonly solo: boolean
  readonly systemPromptOverride?: string
}

const DEFAULT: SessionOptions = {
  sessionId: '',
  disableShellSafeguards: false,
  disableCwdSafeguards: false,
  timezone: null,
  vcsAvailable: true,
  headless: false,
  solo: false,
}

export const SessionOptionsAmbient = Ambient.define<SessionOptions, never>({
  name: 'SessionOptions',
  initial: Effect.succeed(DEFAULT),
})

export function publishSessionOptions(options: SessionOptions) {
  return Effect.gen(function* () {
    const ambientService = yield* AmbientServiceTag
    yield* ambientService.register(SessionOptionsAmbient)
    yield* ambientService.update(SessionOptionsAmbient, options)
  })
}
