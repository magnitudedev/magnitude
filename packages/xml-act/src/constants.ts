export const TASK_OPEN = '<task>'
export const TASK_CLOSE = '</task>'
export const ASSIGN_OPEN = '<assign>'
export const ASSIGN_CLOSE = '</assign>'
export const LENSES_OPEN = '<lenses>'
export const LENSES_CLOSE = '</lenses>'

export const TURN_CONTROL_IDLE = '<idle/>'
export const TURN_CONTROL_FINISH = '<finish/>'

export const AGENT_CREATE_TAG = 'agent-create'
export const TITLE_TAG = 'title'
export const MESSAGE_TAG = 'message'
export const LENS_TAG = 'lens'
export const TASK_TAG = 'task'
export const ASSIGN_TAG = 'assign'
export const REASSIGN_TAG = 'reassign'

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

export const taskOpen = (attrs?: Record<string, string>) => xmlOpen(TASK_TAG, attrs)
export const taskClose = () => xmlClose(TASK_TAG)

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
