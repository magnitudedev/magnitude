import { Option } from 'effect'

export interface CompactToolCallInput {
  toolName: string
  attributes: Record<string, string>
  body: Option.Option<string>
  maxBodyChars?: number
}

const DEFAULT_MAX_BODY_CHARS = 500

function truncateBody(body: string, maxChars: number): string {
  if (body.length <= maxChars) return body
  return `${body.slice(0, maxChars)}... (truncated)`
}

export function renderCompactToolCall(input: CompactToolCallInput): string {
  const { toolName, attributes, body } = input
  const maxBodyChars = input.maxBodyChars ?? DEFAULT_MAX_BODY_CHARS

  const attrs = Object.entries(attributes)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([key, value]) => ` ${key}="${value}"`)
    .join('')

  return Option.match(body, {
    onNone: () => `<${toolName}${attrs}/>`,
    onSome: (b) => {
      if (b.length === 0) return `<${toolName}${attrs}/>`
      const safeBody = truncateBody(b, maxBodyChars)
      return `<${toolName}${attrs}>${safeBody}</${toolName}>`
    },
  })
}
