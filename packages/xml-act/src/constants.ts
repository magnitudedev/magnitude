// Yield target identifiers (used as the variant in <|yield:TARGET|>)
export const YIELD_USER_TARGET = 'user'
export const YIELD_TOOL_TARGET = 'tool'
export const YIELD_WORKER_TARGET = 'worker'
export const YIELD_PARENT_TARGET = 'parent'

/** Role-specific yield target lists (for grammar builder) */
export const LEAD_YIELD_TAGS = [YIELD_USER_TARGET, YIELD_TOOL_TARGET, YIELD_WORKER_TARGET] as const
export const SUBAGENT_YIELD_TAGS = [YIELD_PARENT_TARGET, YIELD_TOOL_TARGET] as const

// Yield format strings
export const YIELD_USER = '<|yield:user|>'
export const YIELD_TOOL = '<|yield:tool|>'
export const YIELD_WORKER = '<|yield:worker|>'
export const YIELD_PARENT = '<|yield:parent|>'

// Stop sequences
export const YIELD_USER_STOP = YIELD_USER
export const YIELD_TOOL_STOP = YIELD_TOOL
export const YIELD_WORKER_STOP = YIELD_WORKER
export const YIELD_PARENT_STOP = YIELD_PARENT

export const LEAD_YIELD_STOP_SEQUENCES = [YIELD_USER_STOP, YIELD_TOOL_STOP, YIELD_WORKER_STOP] as const
export const SUBAGENT_YIELD_STOP_SEQUENCES = [YIELD_PARENT_STOP, YIELD_TOOL_STOP] as const
