
// Yield tag constants (bare tag names)
export const YIELD_USER_TAG = 'yield-user'
export const YIELD_TOOL_TAG = 'yield-tool'
export const YIELD_WORKER_TAG = 'yield-worker'
export const YIELD_PARENT_TAG = 'yield-parent'

// Full XML strings (self-closing)
export const YIELD_USER = `<${YIELD_USER_TAG}/>`
export const YIELD_TOOL = `<${YIELD_TOOL_TAG}/>`
export const YIELD_WORKER = `<${YIELD_WORKER_TAG}/>`
export const YIELD_PARENT = `<${YIELD_PARENT_TAG}/>`

// Stop sequences (one per yield tag)
export const YIELD_USER_STOP = `<${YIELD_USER_TAG}/>`
export const YIELD_TOOL_STOP = `<${YIELD_TOOL_TAG}/>`
export const YIELD_WORKER_STOP = `<${YIELD_WORKER_TAG}/>`
export const YIELD_PARENT_STOP = `<${YIELD_PARENT_TAG}/>`

// Role-specific groupings
export const LEAD_YIELD_STOP_SEQUENCES = [YIELD_USER_STOP, YIELD_TOOL_STOP, YIELD_WORKER_STOP] as const
export const SUBAGENT_YIELD_STOP_SEQUENCES = [YIELD_PARENT_STOP, YIELD_TOOL_STOP] as const
export const LEAD_YIELD_TAGS = [YIELD_USER_TAG, YIELD_TOOL_TAG, YIELD_WORKER_TAG] as const
export const SUBAGENT_YIELD_TAGS = [YIELD_PARENT_TAG, YIELD_TOOL_TAG] as const

// Other constants
export const AGENT_CREATE_TAG = 'agent-create'
export const TITLE_TAG = 'title'
export const MESSAGE_TAG = 'message'
export const LENS_TAG = 'lens'

export const AGENT_CREATE_OPEN_PREFIX = '<agent-create'
export const AGENT_CREATE_CLOSE = '</agent-create>'
export const TITLE_OPEN = '<title>'
export const TITLE_CLOSE = '</title>'
export const MESSAGE_OPEN = '<message>'
export const MESSAGE_CLOSE = '</message>'

export const xmlOpen = (tag: string, attrs?: Record<string, string>) =>
  `<${tag}${attrs ? ` ${Object.entries(attrs).map(([k, v]) => `${k}="${v}"`).join(' ')}` : ''}>`

export const xmlClose = (tag: string) => `</${tag}>`

export const lensOpen = (name: string) => xmlOpen(LENS_TAG, { name })
export const lensClose = () => xmlClose(LENS_TAG)

export const messageOpen = () => xmlOpen(MESSAGE_TAG)
export const messageClose = () => MESSAGE_CLOSE

export const agentCreateOpen = (
  attrs: { id: string; type: string; [key: string]: string },
  options?: { multiline?: boolean },
) =>
  options?.multiline
    ? `${AGENT_CREATE_OPEN_PREFIX}
${Object.entries(attrs)
  .map(([k, v]) => `${k}="${v}"`)
  .join('\n')}
>`
    : xmlOpen(AGENT_CREATE_TAG, attrs)
