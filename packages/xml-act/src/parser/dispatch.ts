import type {
  ParseEvent,
  ParseStack,
  ParserConfig,
  StepResult,
} from './types'
import { NOOP } from './types'
import { stepProse } from './prose'
import { stepThink, stepThinkClosePrefixMatch, stepPendingThinkClose, stepLensOpenPrefixMatch, stepLensTagAttrs } from './think'
import { stepMessageBody, stepMessageOpenPrefixMatch, stepMessageOpenTagTail, stepMessageClosePrefixMatch } from './message'
import { stepToolBody, stepToolClosePrefixMatch, stepChildOpenPrefixMatch, stepChildAttrs, stepChildAttrValue, stepChildUnquotedAttrValue, stepChildBody, stepChildClosePrefixMatch } from './tool-body'
import { stepOpenPrefixMatch, stepClosePrefixMatch, stepTagAttrs, stepTagAttrValue, stepTagUnquotedAttrValue, stepPendingStructuralOpen, stepPendingTopLevelClose } from './top-level'
import { stepFinishBody, stepFinishClosePrefixMatch } from './finish'
import { stepCdata } from './cdata'

export function processChar(state: ParseStack, ch: string, config: ParserConfig): ParseEvent[] {
  const allEvents: ParseEvent[] = []
  let result = dispatch(state, ch, config)

  while (result._tag === 'Reprocess' || result._tag === 'EmitAndReprocess') {
    if (result._tag === 'EmitAndReprocess') allEvents.push(...result.events)
    result = dispatch(state, ch, config)
  }

  if (result._tag === 'Emit') allEvents.push(...result.events)
  return allEvents
}

function dispatch(state: ParseStack, ch: string, config: ParserConfig): StepResult {
  const frame = state[state.length - 1]
  if (!frame) return NOOP

  switch (frame._tag) {
    case 'Prose': return stepProse({ frame, state, ch, config })
    case 'OpenPrefixMatch': return stepOpenPrefixMatch({ frame, state, ch, config })
    case 'ClosePrefixMatch': return stepClosePrefixMatch({ frame, state, ch, config })
    case 'TagAttrs': return stepTagAttrs({ frame, state, ch, config })
    case 'TagAttrValue': return stepTagAttrValue({ frame, state, ch, config })
    case 'TagUnquotedAttrValue': return stepTagUnquotedAttrValue({ frame, state, ch, config })
    case 'Think': return stepThink({ frame, state, ch, config })
    case 'ThinkClosePrefixMatch': return stepThinkClosePrefixMatch({ frame, state, ch, config })
    case 'PendingThinkClose': return stepPendingThinkClose({ frame, state, ch, config })
    case 'LensOpenPrefixMatch': return stepLensOpenPrefixMatch({ frame, state, ch, config })
    case 'LensTagAttrs': return stepLensTagAttrs({ frame, state, ch, config })
    case 'PendingStructuralOpen': return stepPendingStructuralOpen({ frame, state, ch, config })
    case 'PendingTopLevelClose': return stepPendingTopLevelClose({ frame, state, ch, config })
    case 'MessageBody': return stepMessageBody({ frame, state, ch, config })
    case 'MessageOpenPrefixMatch': return stepMessageOpenPrefixMatch({ frame, state, ch, config })
    case 'MessageClosePrefixMatch': return stepMessageClosePrefixMatch({ frame, state, ch, config })
    case 'MessageOpenTagTail': return stepMessageOpenTagTail({ frame, state, ch, config })
    case 'ToolBody': return stepToolBody({ frame, state, ch, config })
    case 'ToolClosePrefixMatch': return stepToolClosePrefixMatch({ frame, state, ch, config })
    case 'ChildOpenPrefixMatch': return stepChildOpenPrefixMatch({ frame, state, ch, config })
    case 'ChildAttrs': return stepChildAttrs({ frame, state, ch, config })
    case 'ChildAttrValue': return stepChildAttrValue({ frame, state, ch, config })
    case 'ChildUnquotedAttrValue': return stepChildUnquotedAttrValue({ frame, state, ch, config })
    case 'ChildBody': return stepChildBody({ frame, state, ch, config })
    case 'ChildClosePrefixMatch': return stepChildClosePrefixMatch({ frame, state, ch, config })
    case 'Cdata': return stepCdata({ frame, state, ch, config })
    case 'FinishBody': return stepFinishBody({ frame, state, ch, config })
    case 'FinishClosePrefixMatch': return stepFinishClosePrefixMatch({ frame, state, ch, config })
    case 'Actions':
    case 'Comms':
    case 'Done':
      return NOOP
  }
}