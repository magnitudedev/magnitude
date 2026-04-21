import { defineStateModel, type BaseState } from '@magnitudedev/tools'
import { viewTool } from '../tools/fs'

export interface FileViewState extends BaseState {
  toolKey: 'fileView'
  path?: string
}

const initial: Omit<FileViewState, 'phase' | 'toolKey'> = {
  path: undefined,
}

export const fileViewModel = defineStateModel('fileView', viewTool)({
  initial,
  reduce: (state, event): FileViewState => {
    switch (event._tag) {
      case 'ToolInputStarted':
        return { ...state, phase: 'streaming' }
      case 'ToolInputFieldChunk':
        return event.field === 'path'
          ? { ...state, phase: 'streaming', path: (state.path ?? '') + event.delta }
          : state
      case 'ToolInputReady':
        return { ...state, phase: 'streaming', path: event.input.path }
      case 'ToolExecutionStarted':
        return { ...state, phase: 'executing' }
      case 'ToolExecutionEnded': {
        switch (event.result._tag) {
          case 'Success':
            return { ...state, phase: 'completed' }
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
