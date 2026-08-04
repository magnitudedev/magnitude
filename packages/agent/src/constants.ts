/**
 * Agent Configuration Constants
 */

/** Default chat name before title generation */
export const DEFAULT_CHAT_NAME = 'New Magnitude Chat'

/** Characters per token upper bound (for truncation budgets — higher = fewer tokens per char = more conservative truncation) */
export const CHARS_PER_TOKEN_UPPER = 4

/** Characters per token lower bound (for compaction estimation — lower = more tokens estimated = safer) */
export const CHARS_PER_TOKEN_LOWER = 3

/** Token budget for the folder structure included in initial session context */
export const FOLDER_TREE_BUDGET_TOKENS = 1_000

/** Maximum VCS status entries included in initial session context */
export const MAX_VCS_STATUS_LINES = 20

/** Max tokens for a resolved ref in an inspect block */
export const TRUNCATION_TOKEN_LIMIT = 10_000

/** Character equivalent of TRUNCATION_TOKEN_LIMIT */
export const TRUNCATION_CHAR_LIMIT = TRUNCATION_TOKEN_LIMIT * CHARS_PER_TOKEN_UPPER


// =============================================================================
// JS-ACT Prose Delimiters
// =============================================================================

/** Opening prose delimiter for JS-ACT string literals */
export const PROSE_DELIM_OPEN = '<raw>'

/** Closing prose delimiter for JS-ACT string literals */
export const PROSE_DELIM_CLOSE = '</raw>'


// =============================================================================
// Context Budget
// =============================================================================

/** Fixed output token reserve subtracted from contextWindow to compute hardCap */
export const OUTPUT_TOKEN_RESERVE = 8_192

// =============================================================================
// Tool Execution
// =============================================================================

/** Maximum tool calls accepted from one agent model response */
export const MAX_TOOL_CALLS_PER_TURN = 10

// =============================================================================
// Compaction
// =============================================================================

/** Maximum number of files the compact() tool will read */
export const COMPACT_MAX_FILES = 10

/** Maximum characters per file in compact() tool output */
export const COMPACT_MAX_FILE_CHARS = 10_000

/** Maximum retry attempts for compaction if agent doesn't call compact() */
export const COMPACTION_MAX_RETRIES = 3

/** Fraction of content to keep as fallback when all compaction retries fail */
export const COMPACTION_FALLBACK_KEEP_RATIO = 0.25

/** Fraction of soft cap to keep as recent messages during compaction */
export const KEEP_MESSAGE_RATIO = 0.1

/** Fraction of messages to trim from compaction input on each retry when input exceeds context window */
export const EMERGENCY_COMPACT_CONTEXT_TRIM_RATIO = 0.2

// =============================================================================
