import { defineStateModel, type BaseState } from '@magnitudedev/tools'
import { writeTool } from '../tools/fs'

export interface FileWriteState extends BaseState {
  toolKey: 'fileWrite'
  path?: string
  body: string
  charCount: number
  lineCount: number
}

const initial: Omit<FileWriteState, 'phase' | 'toolKey'> = {
  path: undefined,
  body: '',
  charCount: 0,
  lineCount: 0,
}

export const fileWriteModel = defineStateModel('fileWrite', writeTool)({
  initial,
  reduce: (state, event): FileWriteState => {
    switch (event._tag) {
      case 'ToolInputStarted':
        return { ...state, phase: 'streaming' }
      case 'ToolInputFieldChunk': {
        if (event.field === 'path') return { ...state, phase: 'streaming', path: (state.path ?? '') + event.delta }
        if (event.field === 'content') {
          const content = state.body + event.delta
          return {
            ...state,
            phase: 'streaming',
            body: content,
            charCount: content.length,
            lineCount: content.length > 0 ? content.split('\n').length : 0,
          }
        }
        return state
      }
      case 'ToolInputReady': {
        const content = event.input.content
        return {
          ...state,
          phase: 'streaming',
          path: event.input.path ?? state.path,
          body: content,
          charCount: content.length,
          lineCount: content.length > 0 ? content.split('\n').length : 0,
        }
      }
      case 'ToolExecutionStarted':
        return { ...state, phase: 'executing' }
      case 'ToolEmission': {
        const v = event.value as { type?: string; path?: string; linesWritten?: number }
        return v.type === 'write_stats'
          ? { ...state, phase: 'executing', path: v.path ?? state.path, lineCount: v.linesWritten ?? state.lineCount }
          : state
      }
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
      case 'ToolInputFieldComplete':
      default:
        return state
    }
  },
})
