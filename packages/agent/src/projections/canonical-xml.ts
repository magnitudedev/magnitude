import { YIELD_USER, YIELD_TOOL } from '@magnitudedev/xml-act'
import type { MessageDestination } from '../events'

export interface ThinkBlock {
  about: string | null
  content: string
}

export interface CanonicalTrace {
  lenses: readonly { name: string; content: string | null }[] | null
  thinkBlocks: ThinkBlock[]
  messages: Array<{ text: string; destination: MessageDestination }>
  toolCalls: Array<{ tagName: string; input: unknown; query: string }>
  turnDecision: 'continue' | 'idle'
}

function attrsToString(attrs: Record<string, string>): string {
  const entries = Object.entries(attrs).sort(([a], [b]) => a.localeCompare(b))
  if (entries.length === 0) return ''
  return entries.map(([k, v]) => ` ${k}="${v}"`).join('')
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

function setByPath(obj: Record<string, unknown>, path: string, value: unknown): void {
  const segments = path.split('.')
  let current = obj
  for (let i = 0; i < segments.length - 1; i++) {
    if (!(segments[i] in current) || typeof current[segments[i]] !== 'object' || current[segments[i]] === null) {
      current[segments[i]] = {}
    }
    current = current[segments[i]] as Record<string, unknown>
  }
  current[segments[segments.length - 1]] = value
}

function serializeTag(name: string, attrs: Record<string, string>, body: string | null, children: string[]): string {
  const attrStr = attrsToString(attrs)
  if (body === null && children.length === 0) {
    return `<${name}${attrStr} />`
  }
  return `<${name}${attrStr}>${children.join('')}${body ?? ''}</${name}>`
}

function serializeToolCall(tagName: string, input: unknown, query: string | null): string {
  const obj = (input && typeof input === 'object') ? input as Record<string, unknown> : {}
  const observeAttr = query ? ` observe="${query}"` : ''

  // Serialize as JSON body for simplicity
  return Object.keys(obj).length === 0
    ? `<${tagName}${observeAttr} />`
    : `<${tagName}${observeAttr}>${JSON.stringify(input)}</${tagName}>`
}

export function serializeCanonicalTurn(
  trace: CanonicalTrace,
): string {
  const parts: string[] = []

  if (trace.lenses !== null) {
    const activeLenses = trace.lenses.filter((lens) => lens.content !== null)
    if (activeLenses.length > 0) {
      const lensLines = activeLenses.map((lens) =>
        `<lens name="${lens.name}">${lens.content}</lens>`
      )
      parts.push(lensLines.join('\n'))
    }
  }

  for (const block of trace.thinkBlocks) {
    const trimmed = block.content.trim()
    if (trimmed.length === 0) continue
    const aboutAttr = block.about ? ` about="${block.about}"` : ''
    parts.push(`<think${aboutAttr}>${trimmed}</think>`)
  }

  if (trace.messages.length > 0) {
    for (const msg of trace.messages) {
      const trimmedText = msg.text.trim()
      const attrs: Record<string, string> = {}
      if (msg.destination.kind === 'worker') attrs.to = msg.destination.taskId
      else if (msg.destination.kind === 'parent') attrs.to = 'parent'
      else if (msg.destination.kind === 'user') attrs.to = 'user'
      parts.push(serializeTag('message', attrs, trimmedText, []))
    }
  }

  if (trace.toolCalls.length > 0) {
    for (const call of trace.toolCalls) {
      parts.push(serializeToolCall(call.tagName, call.input, call.query))
    }
  }

  if (trace.turnDecision === 'idle') {
    parts.push(YIELD_USER)
  } else if (trace.turnDecision === 'continue') {
    parts.push(YIELD_TOOL)
  }
  return parts.join('\n')
}