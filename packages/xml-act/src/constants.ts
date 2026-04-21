// Known MACT structural tag names
/** Tags that require newline enforcement (block-level elements) */
export const TOP_LEVEL_TAGS: ReadonlySet<string> = new Set(['think', 'message', 'invoke', 'yield'])

/** Tags that can appear as bare close `<name>` (Mode 3 leniency) */
export const KNOWN_CLOSE_TAG_NAMES: ReadonlySet<string> = new Set(['think', 'message', 'invoke', 'parameter', 'filter', 'yield'])

/** Tags that produce a stray-close error when unmatched */
export const KNOWN_STRUCTURAL_TAGS: ReadonlySet<string> = new Set(['think', 'message', 'invoke'])

// Yield target identifiers (used as the variant in <|yield:TARGET|>)
export const YIELD_USER_TARGET = 'user'
export const YIELD_INVOKE_TARGET = 'invoke'
export const YIELD_WORKER_TARGET = 'worker'
export const YIELD_PARENT_TARGET = 'parent'

/** Role-specific yield target lists (for grammar builder) */
export const LEAD_YIELD_TAGS = [YIELD_USER_TARGET, YIELD_INVOKE_TARGET, YIELD_WORKER_TARGET] as const
export const SUBAGENT_YIELD_TAGS = [YIELD_PARENT_TARGET, YIELD_INVOKE_TARGET] as const

// Yield format strings
export const YIELD_USER = '<|yield:user|>'
export const YIELD_INVOKE = '<|yield:invoke|>'
export const YIELD_WORKER = '<|yield:worker|>'
export const YIELD_PARENT = '<|yield:parent|>'

// Stop sequences
export const YIELD_USER_STOP = YIELD_USER
export const YIELD_INVOKE_STOP = YIELD_INVOKE
export const YIELD_WORKER_STOP = YIELD_WORKER
export const YIELD_PARENT_STOP = YIELD_PARENT

export const LEAD_YIELD_STOP_SEQUENCES = [YIELD_USER_STOP, YIELD_INVOKE_STOP, YIELD_WORKER_STOP] as const
export const SUBAGENT_YIELD_STOP_SEQUENCES = [YIELD_PARENT_STOP, YIELD_INVOKE_STOP] as const
