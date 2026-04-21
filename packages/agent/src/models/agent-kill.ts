import { defineStateModel, type BaseState } from '@magnitudedev/tools'
import { agentKillTool } from '../tools/agent-tools'

export interface AgentKillState extends BaseState {
  toolKey: 'agentKill'
  agentId?: string
}

const initial: Omit<AgentKillState, 'phase' | 'toolKey'> = {
  agentId: undefined,
}

export const agentKillModel = defineStateModel('agentKill', agentKillTool)({
  initial,
  reduce: (state, event): AgentKillState => {
    switch (event._tag) {
      case 'ToolInputStarted':
        return { ...state, phase: 'streaming' }
      case 'ToolInputFieldChunk':
        return event.field === 'agentId'
          ? { ...state, phase: 'streaming', agentId: (state.agentId ?? '') + event.delta }
          : state
      case 'ToolInputReady':
        return { ...state, phase: 'streaming', agentId: event.input.agentId }
      case 'ToolExecutionStarted':
        return { ...state, phase: 'executing' }
      case 'ToolExecutionEnded': {
        switch (event.result._tag) {
          case 'Success':
            return { ...state, phase: 'completed', agentId: event.result.output.agentId }
          case 'Error':
            return { ...state, phase: 'error' }
          case 'Rejected':
            return { ...state, phase: 'rejected' }
          case 'Interrupted':
            return { ...state, phase: 'interrupted' }
        }
      }
      case 'ToolParseError':
        return { ...state, phase: 'error' }
      case 'ToolEmission':
      case 'ToolInputFieldComplete':
      default:
        return state
    }
  },
})
