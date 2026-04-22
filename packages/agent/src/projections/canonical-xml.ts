import { YIELD_USER, YIELD_INVOKE, deriveParameters } from '@magnitudedev/xml-act'
import type { MessageDestination } from '../events'
import type { ResolvedToolSet } from '../tools/resolved-toolset'
import { buildRegisteredTools } from '../tools/tool-registry'
import { Layer } from 'effect'

export interface ReasonBlock {
  about: string | null
  content: string
}

export interface CanonicalTrace {
  lenses: readonly { name: string; content: string | null }[] | null
  reasonBlocks: ReasonBlock[]
  messages: Array<{ text: string; destination: MessageDestination }>
  toolCalls: Array<{ tagName: string; input: unknown; query: string | null }>
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

function serializeParameter(name: string, value: unknown): string {
  const serialized = (value !== null && typeof value === 'object') ? JSON.stringify(value) : String(value ?? '')
  return `<parameter name="${name}">${serialized}</parameter>`
}

function serializeToolCall(tagName: string, input: unknown, query: string | null, toolSet: ResolvedToolSet): string {
  const obj = (input && typeof input === 'object') ? input as Record<string, unknown> : {}

  const tool = buildRegisteredTools(toolSet, Layer.empty).get(tagName)
  let params: string[]
  if (tool) {
    const paramSchemas = deriveParameters(tool.tool.inputSchema.ast)
    params = []
    for (const [, param] of paramSchemas.parameters) {
      const value = getByPath(obj, param.name)
      if (value !== undefined) {
        params.push(serializeParameter(param.name, value))
      }
    }
  } else {
    // Unknown tool — fall back to object key order
    params = Object.entries(obj).map(([key, value]) => serializeParameter(key, value))
  }

  const filterPart = query ? `<filter>\n${query}\n</filter>` : ''
  const children = [...params, ...(filterPart ? [filterPart] : [])]
  if (children.length === 0) return `<invoke tool="${tagName}"/>`
  return `<invoke tool="${tagName}">\n${children.join('\n')}\n</invoke>`
}

/**
 * Serialize a canonical trace to XML string.
 * toolSet is required to derive parameter schemas — parameters are emitted
 * in schema-defined order for deterministic output.
 */
export function serializeCanonicalTurn(
  trace: CanonicalTrace,
  toolSet: ResolvedToolSet,
): string {
  const parts: string[] = []

  if (trace.lenses !== null) {
    const activeLenses = trace.lenses.filter((lens) => lens.content !== null && lens.content.trim().length > 0)
    for (const lens of activeLenses) {
      parts.push(`<reason about="${lens.name}">\n${lens.content!.trim()}\n</reason>`)
    }
  }

  for (const block of trace.reasonBlocks) {
    const trimmed = block.content.trim()
    if (trimmed.length === 0) continue
    const name = block.about ?? 'reason'
    parts.push(`<reason about="${name}">\n${trimmed}\n</reason>`)
  }

  for (const msg of trace.messages) {
    const trimmedText = msg.text.trim()
    let to: string
    if (msg.destination.kind === 'worker') to = msg.destination.taskId
    else if (msg.destination.kind === 'parent') to = 'parent'
    else to = 'user'
    parts.push(`<message to="${to}">\n${trimmedText}\n</message>`)
  }

  for (const call of trace.toolCalls) {
    parts.push(serializeToolCall(call.tagName, call.input, call.query, toolSet))
  }

  if (trace.turnDecision === 'idle') {
    parts.push(YIELD_USER)
  } else if (trace.turnDecision === 'continue') {
    parts.push(YIELD_INVOKE)
  }
  return parts.join('\n')
}
