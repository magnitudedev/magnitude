import type { MagnitudeSlot } from './model-slots'

/**
 * Agent Configuration Constants
 */

/** Default chat name before title generation */
export const DEFAULT_CHAT_NAME = 'New Chat'

/** Characters per token estimate (conservative) */
export const CHARS_PER_TOKEN = 3

/** Max tokens for a resolved ref in an inspect block */
export const INSPECT_TOKEN_LIMIT = 25_000

/** Character equivalent of INSPECT_TOKEN_LIMIT */
export const INSPECT_CHAR_LIMIT = INSPECT_TOKEN_LIMIT * CHARS_PER_TOKEN


// =============================================================================
// JS-ACT Prose Delimiters
// =============================================================================

/** Opening prose delimiter for JS-ACT string literals */
export const PROSE_DELIM_OPEN = '<raw>'

/** Closing prose delimiter for JS-ACT string literals */
export const PROSE_DELIM_CLOSE = '</raw>'


// =============================================================================
// Compaction
// =============================================================================

/** Default context window size when model doesn't specify one */
export const DEFAULT_CONTEXT_WINDOW = 200_000

/** Ratio of context window at which to trigger compaction proactively (soft cap) */
export const COMPACT_TRIGGER_RATIO = 0.9

/** Fraction of soft cap to keep as recent messages during compaction */
export const KEEP_MESSAGE_RATIO = 0.1

/** Fraction of messages to trim from compaction input on each retry when input exceeds context window */
export const EMERGENCY_COMPACT_CONTEXT_TRIM_RATIO = 0.2

// =============================================================================
// User Presence
// =============================================================================

/** How long the window must be blurred before a return is considered an extended absence (ms) */
export const USER_AWAY_RETURN_THRESHOLD_MS = 60_000
export const USER_PRESENCE_CONFIRM_DELAY_MS = 3_000
export const USER_BLUR_DEBOUNCE_MS = 5_000

// =============================================================================

/** Get context limits for a model slot */
export function getContextLimits(_slot: MagnitudeSlot = 'orchestrator'): { hardCap: number; softCap: number } {
  const hardCap = DEFAULT_CONTEXT_WINDOW
  const softCap = Math.floor(hardCap * COMPACT_TRIGGER_RATIO)
  return { hardCap, softCap }
}
