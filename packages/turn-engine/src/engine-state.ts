/**
 * EngineState — the fold-state of the turn engine.
 *
 * Tracks which tool calls exist, which have outcomes, and which are dead.
 * Replaces ReactorState.
 */

import type { EngineState, ToolOutcome, TurnEngineEvent } from './types'

export function initialEngineState(): EngineState {
  return {
    toolCallMap: new Map(),
    toolOutcomes: new Map(),
    deadToolCalls: new Set(),
    stopped: false,
  }
}

export function foldEngineState(state: EngineState, event: TurnEngineEvent): EngineState {
  switch (event._tag) {
    case 'ToolInputStarted': {
      const toolCallMap = new Map(state.toolCallMap)
      toolCallMap.set(event.toolCallId, event.toolName)
      return { ...state, toolCallMap }
    }

    case 'ToolInputDecodeFailure': {
      const deadToolCalls = new Set(state.deadToolCalls)
      deadToolCalls.add(event.toolCallId)
      const toolOutcomes = new Map(state.toolOutcomes)
      toolOutcomes.set(event.toolCallId, { _tag: 'DecodeFailure' })
      return { ...state, deadToolCalls, toolOutcomes }
    }

    case 'TurnStructureDecodeFailure': {
      return state
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
