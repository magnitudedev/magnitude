
import { YIELD_USER, YIELD_INVOKE, deriveParameters } from '@magnitudedev/xml-act'
import type { MessageDestination } from '../events'
import type { ResolvedToolSet } from '../tools/resolved-toolset'
import { buildRegisteredTools } from '../tools/tool-registry'
import type { ToolDefinition } from '@magnitudedev/tools'
import type { RegisteredTool } from '@magnitudedev/xml-act'
import { Layer } from 'effect'

export interface ThinkBlock {
  about: string | null
  content: string
}

export interface MactTrace {
  lenses: readonly { name: string; content: string | null }[] | null
  thinkBlocks: ThinkBlock[]
  messages: Array<{ text: string; destination: MessageDestination }>
  toolCalls: Array<{ tagName: string; input: unknown; filter: string | null }>
  turnDecision: 'continue' | 'idle'
}

function getByPath(obj: Record<string, unknown>, path: string): unknown {
  const segments = path.split('.')
  let current: unknown = obj
  for (const seg of segments) {
    if (!current || typeof current !== 'object') return undefined
    current = (current as Record<string, unknown>)[seg]
  }
  return current
}

function serializeParameter(name: string, value: unknown, type: 'scalar' | 'json'): string {
  if (type === 'json') {
    return `<|parameter:${name}>${JSON.stringify(value)}<parameter|>`
  }
  // Scalar: coerce to string
  return `<|parameter:${name}>${String(value ?? '')}<parameter|>`
}

function serializeToolCall(
  tagName: string, 
  input: unknown, 
  filter: string | null,
  tool: RegisteredTool | undefined
): string {
  const obj = (input && typeof input === 'object') ? input as Record<string, unknown> : {}

  if (!tool) {
    // Fallback: serialize all object keys as parameters
    const params = Object.entries(obj).map(([key, value]) => {
      const isComplex = value !== null && typeof value === 'object'
      return serializeParameter(key, value, isComplex ? 'json' : 'scalar')
    })
    const filterPart = filter ? `<invoke|filter>\n${filter}\n<filter|>` : '<invoke|>'
    return `<|invoke:${tagName}>\n${params.join('\n')}\n${filterPart}`
  }

  // Derive parameters from tool schema and serialize
  const paramSchemas = deriveParameters(tool.tool.inputSchema.ast)
  
  const params: string[] = []
  for (const [, param] of paramSchemas.parameters) {
    const value = getByPath(obj, param.name)
    if (value !== undefined) {
      params.push(serializeParameter(param.name, value, param.type === 'json' ? 'json' : 'scalar'))
    }
  }

  const paramsBlock = params.length > 0 ? params.join('\n') + '\n' : ''
  const filterPart = filter 
    ? `<invoke|filter>\n${filter}\n<filter|>` 
    : '<invoke|>'

  return `<|invoke:${tagName}>\n${paramsBlock}${filterPart}`
}

function destinationToRecipient(dest: MessageDestination): string {
  switch (dest.kind) {
    case 'user': return 'user'
    case 'parent': return 'parent'
    case 'worker': return dest.taskId
    default: return 'user'
  }
}

export function serializeMactTurn(
  trace: MactTrace,
  toolSet: ResolvedToolSet,
): string {
  const parts: string[] = []

  // Lenses as think blocks with content
  if (trace.lenses !== null) {
    const activeLenses = trace.lenses.filter((lens) => lens.content !== null && lens.content.trim().length > 0)
    for (const lens of activeLenses) {
      parts.push(`<|think:${lens.name}>\n${lens.content?.trim() ?? ''}\n<think|>`)
    }
  }

  // Think blocks
  for (const block of trace.thinkBlocks) {
    const trimmed = block.content.trim()
    if (trimmed.length === 0) continue
    const name = block.about ?? 'think'
    parts.push(`<|think:${name}>\n${trimmed}\n<think|>`)
  }

  // Messages
  if (trace.messages.length > 0) {
    for (const msg of trace.messages) {
      const trimmedText = msg.text.trim()
      const recipient = destinationToRecipient(msg.destination)
      parts.push(`<|message:${recipient}>\n${trimmedText}\n<message|>`)
    }
  }

  // Tool calls
  if (trace.toolCalls.length > 0) {
    for (const call of trace.toolCalls) {
      const tool = buildRegisteredTools(toolSet, Layer.empty).get(call.tagName)
      parts.push(serializeToolCall(call.tagName, call.input, call.filter, tool))
    }
  }

  // Yield
  if (trace.turnDecision === 'idle') {
    parts.push(YIELD_USER)
  } else if (trace.turnDecision === 'continue') {
    parts.push(YIELD_INVOKE)
  }

  return parts.join('\n\n')
}
