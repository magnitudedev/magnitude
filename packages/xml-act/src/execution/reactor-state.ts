import type { ReactorState, ToolOutcome, XmlRuntimeEvent } from '../types'

export function initialReactorState(): ReactorState {
  return {
    toolCallMap: new Map(),

    deadToolCalls: new Set(),
    outputTrees: new Map(),
    stopped: false,
    toolOutcomes: new Map(),
  }
}

export function foldReactorState(state: ReactorState, event: XmlRuntimeEvent): ReactorState {
  switch (event._tag) {
    case 'ToolInputStarted': {
      const toolCallMap = new Map(state.toolCallMap)
      toolCallMap.set(event.toolCallId, event.tagName)
      return { ...state, toolCallMap }
    }

    case 'ToolInputParseError': {
      const deadToolCalls = new Set(state.deadToolCalls)
      deadToolCalls.add(event.toolCallId)
      const toolOutcomes = new Map(state.toolOutcomes)
      toolOutcomes.set(event.toolCallId, { _tag: 'ParseError' })
      return { ...state, deadToolCalls, toolOutcomes }
    }

    case 'ToolExecutionEnded': {
      const toolOutcomes = new Map(state.toolOutcomes)
      const outcome: ToolOutcome = { _tag: 'Completed', result: event.result }
      toolOutcomes.set(event.toolCallId, outcome)
      return { ...state, toolOutcomes }
    }

    case 'TurnEnd':
      return { ...state, stopped: true }

    default:
      return state
  }
}
