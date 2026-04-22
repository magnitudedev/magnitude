// Known XML structural tag names
/** Top-level block tags (reason, message, invoke). Yield tags are self-closing and handled separately. */
export const TOP_LEVEL_TAGS: ReadonlySet<string> = new Set(['reason', 'message', 'invoke'])

/** All known close tag names (structural + parameter/filter children) */
export const KNOWN_CLOSE_TAG_NAMES: ReadonlySet<string> = new Set(['reason', 'message', 'invoke', 'parameter', 'filter'])

/** Tags that produce a stray-close error when unmatched */
export const KNOWN_STRUCTURAL_TAGS: ReadonlySet<string> = new Set(['reason', 'message', 'invoke'])

// Yield target string values (used by parser and ExecutionManager)
export const YIELD_USER_TARGET = 'user' as const
export const YIELD_INVOKE_TARGET = 'invoke' as const
export const YIELD_WORKER_TARGET = 'worker' as const
export const YIELD_PARENT_TARGET = 'parent' as const

// Yield tag names (self-closing XML tags)
export const YIELD_USER = '<yield_user/>'
export const YIELD_INVOKE = '<yield_invoke/>'
export const YIELD_WORKER = '<yield_worker/>'
export const YIELD_PARENT = '<yield_parent/>'

/** Role-specific yield tag name lists (for grammar builder — full tag names without < or />) */
export const LEAD_YIELD_TAGS = ['yield_user', 'yield_invoke', 'yield_worker'] as const
export const SUBAGENT_YIELD_TAGS = ['yield_parent', 'yield_invoke'] as const

// Stop sequences (same as yield strings — model stops on seeing the self-close tag)
export const YIELD_USER_STOP = YIELD_USER
export const YIELD_INVOKE_STOP = YIELD_INVOKE
export const YIELD_WORKER_STOP = YIELD_WORKER
export const YIELD_PARENT_STOP = YIELD_PARENT

export const LEAD_YIELD_STOP_SEQUENCES = [YIELD_USER_STOP, YIELD_INVOKE_STOP, YIELD_WORKER_STOP] as const
export const SUBAGENT_YIELD_STOP_SEQUENCES = [YIELD_PARENT_STOP, YIELD_INVOKE_STOP] as const
