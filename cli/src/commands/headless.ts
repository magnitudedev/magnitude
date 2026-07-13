/**
 * Headless mode runner — temporarily disabled.
 *
 * The CLI is transitioning to a pure SDK/RPC client architecture. Headless mode
 * needs a dedicated daemon-backed persistence design. Until that is in place,
 * attempting to run headless exits with an error.
 */

import type { SessionStart } from '../app'

export interface RunHeadlessOptions {
  debug: boolean
  autopilot: boolean
  initialPrompt?: string
  sessionStart: SessionStart
  disableShellSafeguards: boolean
  disableCwdSafeguards: boolean
  atifPath?: string
  goal?: string
  solo: boolean
  systemOverride?: string
}

export async function runHeadless(_options: RunHeadlessOptions): Promise<number> {
  process.stderr.write('Error: --headless is temporarily disabled.\n')
  return 1
}
