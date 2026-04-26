import {
  TAG_THINK,
  TAG_THINK_CLOSE_ALIAS,
  TAG_MESSAGE,
  TAG_INVOKE,
  TAG_PARAMETER,
  TAG_FILTER,
  MAGNITUDE_PREFIX,
} from '../constants'

// =============================================================================
// GBNF escaping
// =============================================================================

export function escapeGbnfChar(ch: string): string {
  switch (ch) {
    case '"': return '\\"'
    case '\\': return '"\\\\"'
    case '\n': return '"\\n"'
    case '\t': return '"\\t"'
    case '<': return '"<"'
    case '>': return '">"'
    default: return `"${ch}"`
  }
}

export function escapeGbnfCharClass(ch: string): string {
  switch (ch) {
    case ']': return '\\]'
    case '\\': return '\\\\'
    case '^': return '\\^'
    case '-': return '\\-'
    case '\n': return '\\n'
    case '\t': return '\\t'
    default: return ch
  }
}

export function escapeGbnfString(s: string): string {
  return s.replace(/"/g, '\\"').replace(/\\/g, '\\\\')
}

// =============================================================================
// Close tag helpers
// =============================================================================

/** Build a GBNF-quoted close tag string from a tag name, e.g. gbnfCloseTag(TAG_THINK) */
export function gbnfCloseTag(tagName: string): string {
  return `"${escapeGbnfString('</' + tagName + '>')}"`
}

// =============================================================================
// Name sanitization
// =============================================================================

export function sanitizeRuleName(tagName: string): string {
  return `t-${tagName.replace(/[^a-zA-Z0-9]/g, '').toLowerCase()}`
}

export function sanitizeParamName(name: string): string {
  return name.replace(/[^a-zA-Z0-9]/g, '').toLowerCase()
}

// =============================================================================
// Re-exported constants for grammar consumers
// =============================================================================

export {
  TAG_THINK,
  TAG_THINK_CLOSE_ALIAS,
  TAG_MESSAGE,
  TAG_INVOKE,
  TAG_PARAMETER,
  TAG_FILTER,
  MAGNITUDE_PREFIX,
}
