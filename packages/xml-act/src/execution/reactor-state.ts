import type { ReactorState, ToolOutcome, RuntimeEvent } from '../types'

export function initialReactorState(): ReactorState {
  return {
    toolCallMap: new Map(),
    deadToolCalls: new Set(),
    stopped: false,
    toolOutcomes: new Map(),
  }
}

export function foldReactorState(state: ReactorState, event: RuntimeEvent): ReactorState {
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
