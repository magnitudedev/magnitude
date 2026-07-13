import { defineStateModel } from '@magnitudedev/harness'
import { Option } from 'effect'
import { grepTool } from '../tools/fs'
import { FileSearchStateSchema, SearchMatchSchema, type FileSearchState, type SearchMatch } from './tool-state'

export { FileSearchStateSchema, SearchMatchSchema, type FileSearchState, type SearchMatch } from './tool-state'

const initial: Omit<FileSearchState, 'phase' | 'errorMessage'> = {
  pattern: Option.none(),
  path: Option.none(),
  glob: Option.none(),
  limit: Option.none(),
  matches: [],
  matchCount: 0,
  fileCount: 0,
  errorDetail: Option.none(),
}

export const fileSearchModel = defineStateModel(grepTool)({
  state: FileSearchStateSchema,
  initial,
  reduce: (state, event): FileSearchState => {
    switch (event._tag) {
      case 'ToolInputStarted':
        return { ...state, phase: 'streaming' }
      case 'ToolInputFieldChunk':
        if (event.field === 'pattern') return { ...state, phase: 'streaming', pattern: Option.some(Option.getOrElse(state.pattern, () => '') + event.delta) }
        if (event.field === 'path') return { ...state, phase: 'streaming', path: Option.some(Option.getOrElse(state.path, () => '') + event.delta) }
        if (event.field === 'glob') return { ...state, phase: 'streaming', glob: Option.some(Option.getOrElse(state.glob, () => '') + event.delta) }
        return state
      case 'ToolInputReady':
        return state
      case 'ToolExecutionStarted':
        return {
          ...state,
          phase: 'executing',
          pattern: typeof event.input.pattern === 'string' ? Option.some(event.input.pattern) : state.pattern,
          path: typeof event.input.path === 'string' ? Option.some(event.input.path) : state.path,
          glob: typeof event.input.glob === 'string' ? Option.some(event.input.glob) : state.glob,
          limit: typeof event.input.limit === 'number' ? Option.some(event.input.limit) : state.limit,
        }
      case 'ToolExecutionEnded': {
        switch (event.result._tag) {
          case 'Success': {
            const matches = [...event.result.output]
            return {
              ...state,
              phase: 'completed',
              matches,
              matchCount: matches.length,
              fileCount: new Set(matches.map((m) => m.file)).size,
            }
          }
          case 'Error':
            return { ...state, phase: 'error', errorDetail: Option.some(event.result.error.message) }
          case 'Denied':
            return { ...state, phase: 'rejected' }
          case 'Interrupted':
            return { ...state, phase: 'interrupted' }
          default:
            return state
        }
      }
      case 'ToolInputRejected':
        return { ...state, phase: 'error', errorDetail: Option.some(event.issue.message) }
      case 'ToolEmission':
      case 'ToolInputFieldComplete':
      default:
        return state
    }
  },
})
