import type { ToolResult, ContentPart } from './types'

// =============================================================================
// renderToolOutput
//
// Converts a ToolResult into ContentPart[] for memory/LLM consumption.
//
// Format (for Success with object output):
//   <fieldName>scalar value, raw and unescaped</fieldName>
//   <fieldName>{"json":"for non-scalar values"}</fieldName>
//
// No outer wrapper — the chat template or codec provides the boundary.
//
// Edge case note: if a scalar string field contains the literal text
// `</fieldName>`, the output will look ambiguous to an XML parser but the
// LLM treats it as plain text. We intentionally do not escape this — the
// design contract explicitly punts on it. Keep in mind if issues arise.
// =============================================================================

function isImageOutput(output: unknown): output is ContentPart & { type: 'image' } {
  return (
    typeof output === 'object' &&
    output !== null &&
    (output as Record<string, unknown>).type === 'image' &&
    typeof (output as Record<string, unknown>).base64 === 'string' &&
    typeof (output as Record<string, unknown>).mediaType === 'string'
  )
}

function isScalar(value: unknown): value is string | number | boolean | null {
  return value === null || typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean'
}

/**
 * Render a single field value inside a tag.
 * - Scalars: raw String(value), placed literally (multi-line strings expand).
 * - Non-scalars: JSON.stringify.
 * - Undefined: field is omitted by the caller.
 *
 * For string scalars that contain newlines, we add a leading and trailing
 * newline inside the tag to keep readability:
 *   <stdout>
 *   line1
 *   line2
 *   </stdout>
 * Single-line strings are inline: <mode>completed</mode>
 */
function renderField(name: string, value: unknown): string {
  if (!isScalar(value)) {
    return `<${name}>${JSON.stringify(value)}</${name}>`
  }
  const raw = String(value)
  if (raw.includes('\n')) {
    return `<${name}>\n${raw}\n</${name}>`
  }
  return `<${name}>${raw}</${name}>`
}

function renderObjectOutput(output: Record<string, unknown>): string {
  const lines: string[] = []
  for (const [key, value] of Object.entries(output)) {
    // Omit undefined fields entirely
    if (value === undefined) continue
    lines.push(renderField(key, value))
  }
  return lines.join('\n')
}

function renderRejection(rejection: unknown): string {
  const inner = isScalar(rejection) ? String(rejection) : JSON.stringify(rejection)
  return `<rejected>${inner}</rejected>`
}

/**
 * Convert a ToolResult into ContentPart[] for memory/LLM consumption.
 *
 * Rules:
 * - Error → `<error>message</error>` (single text part)
 * - Rejected → `<rejected>reason</rejected>` (single text part; non-string rejection JSON-encoded)
 * - Interrupted → `<interrupted/>` (single text part)
 * - Success with image-shaped output → single image ContentPart (bypasses field format)
 * - Success with undefined/void output → `(no output)` (single text part)
 * - Success with scalar output → raw String(output) (single text part)
 * - Success with array output → JSON.stringify(output) (single text part)
 * - Success with object output → one `<field>value</field>` line per top-level field
 */
export function renderToolOutput(result: ToolResult): readonly ContentPart[] {
  switch (result._tag) {
    case 'Error':
      return [{ type: 'text', text: `<error>${result.error}</error>` }]

    case 'Rejected':
      return [{ type: 'text', text: renderRejection(result.rejection) }]

    case 'Interrupted':
      return [{ type: 'text', text: '<interrupted/>' }]

    case 'Success': {
      const { output } = result

      // Undefined / void
      if (output === undefined) {
        return [{ type: 'text', text: '(no output)' }]
      }

      // Image bypass
      if (isImageOutput(output)) {
        return [{ type: 'image', base64: output.base64, mediaType: output.mediaType, width: output.width, height: output.height }]
      }

      // Scalar
      if (isScalar(output)) {
        return [{ type: 'text', text: String(output) }]
      }

      // Array
      if (Array.isArray(output)) {
        return [{ type: 'text', text: JSON.stringify(output) }]
      }

      // Object
      if (typeof output === 'object') {
        const text = renderObjectOutput(output as Record<string, unknown>)
        return [{ type: 'text', text }]
      }

      // Fallback — unknown shape
      return [{ type: 'text', text: JSON.stringify(output) }]
    }
  }
}
