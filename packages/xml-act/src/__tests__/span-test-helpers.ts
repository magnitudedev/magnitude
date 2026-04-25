import type { SourcePos, SourceSpan, Token } from '../types'

export const ZERO_POS: SourcePos = { offset: 0, line: 1, col: 1 }
export const ZERO_SPAN: SourceSpan = { start: ZERO_POS, end: ZERO_POS }

export function stripTokenSpan(token: Token): unknown {
  const { span: _span, ...rest } = token
  return rest
}

export function stripTokenSpans(tokens: readonly Token[]): unknown[] {
  return tokens.map(stripTokenSpan)
}
