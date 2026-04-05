import { LENSES_CLOSE, LENSES_OPEN, TURN_CONTROL_IDLE, type XmlTagBinding } from '@magnitudedev/xml-act'
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

function serializeToolCall(tagName: string, input: unknown, query: string, binding: XmlTagBinding | undefined): string {
  const obj = (input && typeof input === 'object') ? input as Record<string, unknown> : {}

  if (!binding) {
    return Object.keys(obj).length === 0
      ? `<${tagName} observe="${query}" />`
      : `<${tagName} observe="${query}">${JSON.stringify(input)}</${tagName}>`
  }

  const attrs: Record<string, string> = { observe: query }
  if (binding.attributes) {
    for (const attr of binding.attributes) {
      const value = getByPath(obj, attr.field)
      if (value !== undefined && value !== null) attrs[attr.attr] = String(value)
    }
  }

  let body: string | null = null
  if (binding.body) {
    const val = obj[binding.body]
    if (val !== undefined && val !== null) body = String(val)
  }

  const children: string[] = []

  if (binding.childTags) {
    for (const childTag of binding.childTags) {
      const val = getByPath(obj, childTag.field)
      if (val !== undefined && val !== null) {
        children.push(`<${childTag.tag}>${String(val)}</${childTag.tag}>`)
      }
    }
  }

  if (binding.children) {
    for (const child of binding.children) {
      const raw = getByPath(obj, String(child.field))
      if (!Array.isArray(raw)) continue
      const childTag = child.tag ?? String(child.field)
      for (const item of raw) {
        const itemObj = (item && typeof item === 'object') ? item as Record<string, unknown> : {}
        const childAttrs: Record<string, string> = {}
        if (child.attributes) {
          for (const a of child.attributes) {
            const v = itemObj[a.field]
            if (v !== undefined && v !== null) childAttrs[a.attr] = String(v)
          }
        }
        let childBody: string | null = null
        if (child.body) {
          const v = itemObj[child.body]
          if (v !== undefined && v !== null) childBody = String(v)
        }
        children.push(serializeTag(childTag, childAttrs, childBody, []))
      }
    }
  }

  if (binding.childRecord) {
    const recordRaw = getByPath(obj, String(binding.childRecord.field))
    if (recordRaw && typeof recordRaw === 'object') {
      const record = recordRaw as Record<string, unknown>
      for (const key of Object.keys(record).sort((a, b) => a.localeCompare(b))) {
        const value = record[key]
        children.push(
          `<${binding.childRecord.tag} ${binding.childRecord.keyAttr}="${key}">${String(value ?? '')}</${binding.childRecord.tag}>`
        )
      }
    }
  }

  return serializeTag(tagName, attrs, body, children)
}

export function serializeCanonicalTurn(
  trace: CanonicalTrace,
  bindings: Map<string, XmlTagBinding>,
): string {
  const parts: string[] = []

  if (trace.lenses !== null) {
    const activeLenses = trace.lenses.filter((lens) => lens.content !== null)
    if (activeLenses.length > 0) {
      const lensLines = activeLenses.map((lens) =>
        `<lens name="${lens.name}">${lens.content}</lens>`
      )
      parts.push(`${LENSES_OPEN}\n${lensLines.join('\n')}\n${LENSES_CLOSE}`)
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
      parts.push(serializeTag('message', attrs, trimmedText, []))
    }
  }

  if (trace.toolCalls.length > 0) {
    for (const call of trace.toolCalls) {
      parts.push(serializeToolCall(call.tagName, call.input, call.query, bindings.get(call.tagName)))
    }
  }

  if (trace.turnDecision === 'idle') {
    parts.push(TURN_CONTROL_IDLE)
  }
  return parts.join('\n')
}