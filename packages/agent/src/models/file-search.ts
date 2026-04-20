import { defineStateModel, type BaseState } from '@magnitudedev/tools'
import { grepTool } from '../tools/fs'

type SearchMatch = { file: string; match: string }

export interface FileSearchState extends BaseState {
  toolKey: 'fileSearch'
  pattern?: string
  path?: string
  glob?: string
  limit?: number
  matches: SearchMatch[]
  matchCount: number
  fileCount: number
  errorDetail?: string
}

const initial: Omit<FileSearchState, 'phase' | 'toolKey'> = {
  pattern: undefined,
  path: undefined,
  glob: undefined,
  limit: undefined,
  matches: [],
  matchCount: 0,
  fileCount: 0,
  errorDetail: undefined,
}

export const fileSearchModel = defineStateModel('fileSearch', grepTool)({
  initial,
  reduce: (state, event): FileSearchState => {
    switch (event._tag) {
      case 'ToolInputStarted':
        return { ...state, phase: 'streaming' }
      case 'ToolInputFieldChunk':
        if (event.field === 'pattern') return { ...state, phase: 'streaming', pattern: (state.pattern ?? '') + event.delta }
        if (event.field === 'path') return { ...state, phase: 'streaming', path: (state.path ?? '') + event.delta }
        if (event.field === 'glob') return { ...state, phase: 'streaming', glob: (state.glob ?? '') + event.delta }
        return state
      case 'ToolInputReady':
        return {
          ...state,
          phase: 'streaming',
          pattern: event.input.pattern,
          path: event.input.path,
          glob: event.input.glob,
          limit: event.input.limit,
        }
      case 'ToolExecutionStarted':
        return { ...state, phase: 'executing' }
      case 'ToolExecutionEnded': {
        switch (event.result._tag) {
          case 'Success': {
            const matches = [...(event.result.output as SearchMatch[])]
            return {
              ...state,
              phase: 'completed',
              matches,
              matchCount: matches.length,
              fileCount: new Set(matches.map((m) => m.file)).size,
            }
          }
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
