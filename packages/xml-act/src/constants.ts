// Known XML structural tag names
/** Top-level block tags (reason, message, invoke). Yield tags are self-closing and handled separately. */
export const TOP_LEVEL_TAGS: ReadonlySet<string> = new Set(['magnitude:reason', 'magnitude:message', 'magnitude:invoke'])

/** All known close tag names (structural + parameter/filter children) */
export const KNOWN_CLOSE_TAG_NAMES: ReadonlySet<string> = new Set(['magnitude:reason', 'magnitude:message', 'magnitude:invoke', 'magnitude:parameter', 'magnitude:filter', 'magnitude:escape'])

// Individual tag name constants
export const TAG_REASON = 'magnitude:reason' as const
export const TAG_MESSAGE = 'magnitude:message' as const
export const TAG_INVOKE = 'magnitude:invoke' as const
export const TAG_PARAMETER = 'magnitude:parameter' as const
export const TAG_FILTER = 'magnitude:filter' as const
export const TAG_ESCAPE = 'magnitude:escape' as const

export const ESCAPE_TAG = TAG_ESCAPE

/** Tags that produce a stray-close error when unmatched */
export const KNOWN_STRUCTURAL_TAGS: ReadonlySet<string> = new Set(['magnitude:reason', 'magnitude:message', 'magnitude:invoke', 'magnitude:escape'])

// Yield target string values (used by parser and ExecutionManager)
export const YIELD_USER_TARGET = 'user' as const
export const YIELD_INVOKE_TARGET = 'invoke' as const
export const YIELD_WORKER_TARGET = 'worker' as const
export const YIELD_PARENT_TARGET = 'parent' as const

// Yield tag names (self-closing XML tags)
export const YIELD_USER = '<magnitude:yield_user/>'
export const YIELD_INVOKE = '<magnitude:yield_invoke/>'
export const YIELD_WORKER = '<magnitude:yield_worker/>'
export const YIELD_PARENT = '<magnitude:yield_parent/>'

/** Role-specific yield tag name lists (for grammar builder — full tag names without < or />) */
export const LEAD_YIELD_TAGS = ['magnitude:yield_user', 'magnitude:yield_invoke', 'magnitude:yield_worker'] as const
export const SUBAGENT_YIELD_TAGS = ['magnitude:yield_parent', 'magnitude:yield_invoke'] as const

// Stop sequences (same as yield strings — model stops on seeing the self-close tag)
export const YIELD_USER_STOP = YIELD_USER
export const YIELD_INVOKE_STOP = YIELD_INVOKE
export const YIELD_WORKER_STOP = YIELD_WORKER
export const YIELD_PARENT_STOP = YIELD_PARENT

export const LEAD_YIELD_STOP_SEQUENCES = [YIELD_USER_STOP, YIELD_INVOKE_STOP, YIELD_WORKER_STOP] as const
export const SUBAGENT_YIELD_STOP_SEQUENCES = [YIELD_PARENT_STOP, YIELD_INVOKE_STOP] as const