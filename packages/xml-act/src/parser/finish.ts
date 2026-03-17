import type { ParseStack, ParserConfig, StepResult } from './types'
import { emit, NOOP } from './types'
import { advancePrefixMatch } from './prefix-match'

type FinishBodyFrame = Extract<ParseStack[number], { _tag: 'FinishBody' }>
type FinishClosePrefixMatchFrame = Extract<ParseStack[number], { _tag: 'FinishClosePrefixMatch' }>

export function stepFinishBody({ frame, state, ch, config }: { frame: FinishBodyFrame; state: ParseStack; ch: string; config: ParserConfig }): StepResult {
  if (frame.pendingLt) {
    frame.pendingLt = false
    if (ch === '/') {
      state[state.length - 1] = {
        _tag: 'FinishClosePrefixMatch',
        body: frame.body,
        prefix: { candidates: [config.keywords.finish], matched: '', raw: '</' },
      }
      return NOOP
    }
    frame.body += '<' + ch
    return NOOP
  }
  if (ch === '<') { frame.pendingLt = true; return NOOP }
  frame.body += ch
  return NOOP
}

export function stepFinishClosePrefixMatch({ frame, state, ch, config }: { frame: FinishClosePrefixMatchFrame; state: ParseStack; ch: string; config: ParserConfig }): StepResult {
  const advanced = advancePrefixMatch(frame.prefix, ch)

  if (advanced._tag === 'Continue') {
    frame.prefix = { candidates: advanced.candidates, matched: advanced.matched, raw: advanced.raw }
    return NOOP
  }

  if (advanced._tag === 'NoMatch') {
    state[state.length - 1] = { _tag: 'FinishBody', body: frame.body + advanced.literal, pendingLt: false }
    return NOOP
  }

  // Matched </finish>
  if (advanced.delimiter === '>') {
    state.pop()
    const evidence = frame.body.trim() || undefined
    state.push({ _tag: 'Done' })
    return emit({ _tag: 'TurnControl', decision: 'finish', evidence: frame.body.trim() })
  }

  // Delimiter is whitespace or / -- treat as literal
  state[state.length - 1] = { _tag: 'FinishBody', body: frame.body + advanced.raw, pendingLt: false }
  return NOOP
}