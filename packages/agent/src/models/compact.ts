import { defineStateModel } from '@magnitudedev/harness'
import { Option } from 'effect'
import { compactTool } from '../tools/compact'
import { CompactStateSchema, type CompactState } from './tool-state'

export { CompactStateSchema, type CompactState } from './tool-state'

export const compactModel = defineStateModel(compactTool)({
  state: CompactStateSchema,
  initial: {},
  reduce: (state, event) => {
    switch (event._tag) {
      case 'ToolInputStarted':
        return { ...state, phase: 'streaming' as const }
      case 'ToolExecutionStarted':
        return { ...state, phase: 'executing' as const }
      case 'ToolExecutionEnded':
        switch (event.result._tag) {
          case 'Success': return { ...state, phase: 'completed' as const }
          case 'Error': return { ...state, phase: 'error' as const, errorMessage: Option.some(event.result.error.message) }
          case 'Denied': return { ...state, phase: 'rejected' as const, errorMessage: Option.some(String(event.result.denial)) }
          case 'Interrupted': return { ...state, phase: 'interrupted' as const }
          default: return { ...state, phase: 'error' as const }
        }
      case 'ToolInputRejected':
        return { ...state, phase: 'error' as const, errorMessage: Option.some(event.issue.message) }
      default:
        return state
    }
  },
})
