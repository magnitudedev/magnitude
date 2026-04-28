import { defineStateModel } from '@magnitudedev/tools'
import { messageWorkerTool } from '../tools/agent-communication'

export const messageWorkerModel = defineStateModel('messageWorker', messageWorkerTool)({
  initial: {},
  reduce: (state, event) => {
    switch (event._tag) {
      case 'ToolInputStarted':   return { ...state, phase: 'streaming' as const }
      case 'ToolExecutionStarted': return { ...state, phase: 'executing' as const }
      case 'ToolExecutionEnded':
        return { ...state, phase: event.result._tag === 'Success' ? 'completed' as const : 'error' as const }
      default: return state
    }
  },
})
