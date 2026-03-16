// =============================================================================
// Keyword Sets — structural tag names for actions and think blocks.
//
// The "alt" set uses prefixed names (`magniactions`, `magnithink`) to avoid
// collisions when the agent is editing its own source code.
// =============================================================================

export interface KeywordSet {
  readonly actions: string
  readonly think: string
  readonly thinking: string
  readonly lenses: string
  readonly comms: string
}

export const TURN_CONTROL_NEXT = 'next'
export const TURN_CONTROL_YIELD = 'yield'
export const TURN_CONTROL_FINISH = 'finish'

const STANDARD: KeywordSet = {
  actions: 'actions',
  think: 'think',
  thinking: 'thinking',
  lenses: 'lenses',
  comms: 'comms',
}

const ALT: KeywordSet = {
  actions: 'tooluse',
  think: 'reason',
  thinking: 'reasoning',
  lenses: 'lenses',
  comms: 'comms',
}

let active: KeywordSet = STANDARD

/** Switch to the alternate keyword set (for self-development). */
export function useAltKeywords(): void {
  active = ALT
}

/** Switch back to the standard keyword set. */
export function useDefaultKeywords(): void {
  active = STANDARD
}

/** Get the currently active keyword set. */
export function getKeywords(): KeywordSet {
  return active
}

/** Convenience — open/close tags derived from the active set. */
export function actionsTagOpen(): string {
  return `<${active.actions}>`
}

export function actionsTagClose(): string {
  return `</${active.actions}>`
}

export function thinkTagOpen(): string {
  return `<${active.think}>`
}

export function thinkTagClose(): string {
  return `</${active.think}>`
}

export function commsTagOpen(): string {
  return `<${active.comms}>`
}

export function commsTagClose(): string {
  return `</${active.comms}>`
}
