import { defineStateModel, type BaseState } from '@magnitudedev/tools'
import { readTool } from '../tools/fs'

export interface FileReadState extends BaseState {
  toolKey: 'fileRead'
  path?: string
  lineCount?: number
  errorDetail?: string
}

const initial: Omit<FileReadState, 'phase' | 'toolKey'> = {
  path: undefined,
  lineCount: undefined,
  errorDetail: undefined,
}

export const fileReadModel = defineStateModel('fileRead', readTool)({
  initial,
  reduce: (state, event): FileReadState => {
    switch (event._tag) {
      case 'ToolInputStarted':
        return { ...state, phase: 'streaming', errorDetail: undefined }
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
            return { ...state, phase: 'completed', lineCount: (event.result.output as string).split('\n').length }
          case 'Error':
            return { ...state, phase: 'error', errorDetail: event.result.error }
          case 'Rejected':
            return { ...state, phase: 'rejected' }
          case 'Interrupted':
            return { ...state, phase: 'interrupted' }
        }
      }
      case 'ToolInputParseError':
        return { ...state, phase: 'error', errorDetail: event.error.detail }
      case 'ToolEmission':
      case 'ToolInputFieldComplete':
      default:
        return state
    }
  },
})
