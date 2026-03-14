import type {
  ParseEvent,
  ParseStack,
  ParserConfig,
  StepResult,
} from './types'
import { NOOP } from './types'
import { stepProse } from './prose'
import { stepThink, stepThinkCloseTag, stepPendingThinkClose, stepLensTagName, stepLensTagAttrs } from './think'
import { stepMessageBody, stepMessageBodyOpenTag, stepMessageCloseTag } from './message'
import { stepToolBody, stepToolCloseTag, stepChildTagName, stepChildAttrs, stepChildAttrValue, stepChildUnquotedAttrValue, stepChildBody, stepChildCloseTag } from './tool-body'
import { stepTagName, stepTopLevelCloseTag, stepTagAttrs, stepTagAttrValue, stepTagUnquotedAttrValue, stepPendingStructuralOpen, stepPendingTopLevelClose } from './top-level'
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
    case 'TagName': return stepTagName({ frame, state, ch, config })
    case 'TopLevelCloseTag': return stepTopLevelCloseTag({ frame, state, ch, config })
    case 'TagAttrs': return stepTagAttrs({ frame, state, ch, config })
    case 'TagAttrValue': return stepTagAttrValue({ frame, state, ch, config })
    case 'TagUnquotedAttrValue': return stepTagUnquotedAttrValue({ frame, state, ch, config })
    case 'Think': return stepThink({ frame, state, ch, config })
    case 'ThinkCloseTag': return stepThinkCloseTag({ frame, state, ch, config })
    case 'PendingThinkClose': return stepPendingThinkClose({ frame, state, ch, config })
    case 'LensTagName': return stepLensTagName({ frame, state, ch, config })
    case 'LensTagAttrs': return stepLensTagAttrs({ frame, state, ch, config })
    case 'PendingStructuralOpen': return stepPendingStructuralOpen({ frame, state, ch, config })
    case 'PendingTopLevelClose': return stepPendingTopLevelClose({ frame, state, ch, config })
    case 'MessageBody': return stepMessageBody({ frame, state, ch, config })
    case 'MessageBodyOpenTag': return stepMessageBodyOpenTag({ frame, state, ch, config })
    case 'MessageCloseTag': return stepMessageCloseTag({ frame, state, ch, config })
    case 'ToolBody': return stepToolBody({ frame, state, ch, config })
    case 'ToolCloseTag': return stepToolCloseTag({ frame, state, ch, config })
    case 'ChildTagName': return stepChildTagName({ frame, state, ch, config })
    case 'ChildAttrs': return stepChildAttrs({ frame, state, ch, config })
    case 'ChildAttrValue': return stepChildAttrValue({ frame, state, ch, config })
    case 'ChildUnquotedAttrValue': return stepChildUnquotedAttrValue({ frame, state, ch, config })
    case 'ChildBody': return stepChildBody({ frame, state, ch, config })
    case 'ChildCloseTag': return stepChildCloseTag({ frame, state, ch, config })
    case 'Cdata': return stepCdata({ frame, state, ch, config })
    case 'Actions':
    case 'Inspect':
    case 'Comms':
    case 'Done':
      return NOOP
  }
}