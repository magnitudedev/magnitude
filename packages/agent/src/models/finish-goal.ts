export type { Phase, BaseState } from '@magnitudedev/harness'
import { defineStateModel } from '@magnitudedev/harness'
import { Option } from 'effect'
import { finishGoalTool } from '../tools/goal'
import { FinishGoalStateSchema, type FinishGoalState } from './tool-state'

export { FinishGoalStateSchema, type FinishGoalState } from './tool-state'

function append(current: Option.Option<string>, delta: string): Option.Option<string> {
  return Option.some(`${Option.getOrElse(current, () => '')}${delta}`)
}

export const finishGoalModel = defineStateModel(finishGoalTool)({
  state: FinishGoalStateSchema,
  initial: {
    evidence: Option.none(),
    goalId: Option.none(),
  },
  reduce: (state, event): FinishGoalState => {
    switch (event._tag) {
      case 'ToolInputStarted':
        return { ...state, phase: 'streaming', errorMessage: Option.none() }
      case 'ToolInputFieldChunk':
        return event.field === 'evidence'
          ? { ...state, phase: 'streaming', evidence: append(state.evidence, event.delta) }
          : state
      case 'ToolExecutionStarted':
        return {
          ...state,
          phase: 'executing',
          evidence: typeof event.input.evidence === 'string' ? Option.some(event.input.evidence) : state.evidence,
          errorMessage: Option.none(),
        }
      case 'ToolExecutionEnded':
        switch (event.result._tag) {
          case 'Success':
            return {
              ...state,
              phase: 'completed',
              goalId: Option.some(event.result.output.goalId),
              evidence: Option.some(event.result.output.evidence),
            }
          case 'Error':
            return { ...state, phase: 'error', errorMessage: Option.some(event.result.error.message) }
          case 'Denied':
            return { ...state, phase: 'rejected' }
          case 'Interrupted':
            return { ...state, phase: 'interrupted' }
          case 'InputRejected':
            return { ...state, phase: 'error', errorMessage: Option.some(event.result.issue.message) }
        }
      case 'ToolInputRejected':
        return { ...state, phase: 'error', errorMessage: Option.some(event.issue.message) }
      case 'ToolInputReady':
      case 'ToolInputFieldComplete':
      case 'ToolEmission':
      default:
        return state
    }
  },
})
