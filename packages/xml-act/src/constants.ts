export const TURN_CONTROL_CONTINUE_TAG = 'continue'
export const TURN_CONTROL_IDLE_TAG = 'idle'
export const TURN_CONTROL_FINISH_TAG = 'finish'

export const TURN_CONTROL_CONTINUE = `<${TURN_CONTROL_CONTINUE_TAG}/>`
export const TURN_CONTROL_IDLE = `<${TURN_CONTROL_IDLE_TAG}/>`
export const TURN_CONTROL_FINISH = `<${TURN_CONTROL_FINISH_TAG}/>`

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
