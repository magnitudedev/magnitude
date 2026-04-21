/**
 * Token resolution — context-aware structural vs content classification,
 * and raw text reconstruction for non-structural tokens.
 */

import type { Token } from '../types'
import type { Frame } from './types'

/** Structural tags that produce an error when appearing as stray closes */
export const KNOWN_STRUCTURAL_TAGS: ReadonlySet<string> = new Set(['think', 'message', 'invoke'])

/**
 * Reconstruct the raw text of a token for appending as literal content
 * when the token is not structural in the current context.
 */
export function tokenRaw(token: Token): string {
  switch (token._tag) {
    case 'Open':
      return `<|${token.name}${token.variant ? ':' + token.variant : ''}>`
    case 'Close':
      return `<${token.name}${token.pipe ? '|' + token.pipe : '|'}>`
    case 'SelfClose':
      return `<|${token.name}${token.variant ? ':' + token.variant : ''}|>`
    case 'Parameter':
      return `<|parameter:${token.name}>`
    case 'ParameterClose':
      return '<parameter|>'
    case 'Content':
      return token.text
  }
}

/**
 * Resolve a token against the current top frame.
 * Returns 'structural' if the token should be handled as a parser event,
 * or 'content' if it should be appended as literal text.
 */
export function resolveToken(token: Token, top: Frame): 'structural' | 'content' {
  switch (token._tag) {
    case 'Open':
      return top.validTags.has(token.name) ? 'structural' : 'content'
    case 'Close':
      // Piped close (filter start) only valid in invoke
      if (token.pipe) return top.type === 'invoke' ? 'structural' : 'content'
      // Close tags are structural only if the current frame matches the tag
      // (e.g., </think> only structural in a ThinkFrame, not in Prose even though think is in prose's validTags)
      switch (token.name) {
        case 'think': return top.type === 'think' ? 'structural' : 'content'
        case 'message': return top.type === 'message' ? 'structural' : 'content'
        case 'invoke': return top.type === 'invoke' ? 'structural' : 'content'
        case 'parameter': return top.type === 'parameter' ? 'structural' : 'content'
        case 'filter': return top.type === 'filter' ? 'structural' : 'content'
        default: return 'content'
      }
    case 'SelfClose':
      // yield only valid in prose
      return (top.type === 'prose' && token.name === 'yield') ? 'structural' : 'content'
    case 'Parameter':
      // Parameter tokens only valid in invoke
      return top.type === 'invoke' ? 'structural' : 'content'
    case 'ParameterClose':
      // ParameterClose only valid in parameter frame
      return top.type === 'parameter' ? 'structural' : 'content'
    case 'Content':
      return 'structural'
  }
}
