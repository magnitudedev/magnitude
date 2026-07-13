import { defineStateModel } from '@magnitudedev/harness'
import { Option } from 'effect'
import { treeTool } from '../tools/fs'
import { FileTreeStateSchema, TreeEntrySchema, type FileTreeState, type TreeEntry } from './tool-state'

export { FileTreeStateSchema, TreeEntrySchema, type FileTreeState, type TreeEntry } from './tool-state'

const initial: Omit<FileTreeState, 'phase' | 'errorMessage'> = {
  path: Option.none(),
  entries: [],
  fileCount: 0,
  dirCount: 0,
  errorDetail: Option.none(),
}

export const fileTreeModel = defineStateModel(treeTool)({
  state: FileTreeStateSchema,
  initial,
  reduce: (state, event): FileTreeState => {
    switch (event._tag) {
      case 'ToolInputStarted':
        return { ...state, phase: 'streaming' }
      case 'ToolInputFieldChunk':
        return event.field === 'path'
          ? { ...state, phase: 'streaming', path: Option.some(Option.getOrElse(state.path, () => '') + event.delta) }
          : state
      case 'ToolInputReady':
        return state
      case 'ToolExecutionStarted':
        return {
          ...state,
          phase: 'executing',
          path: typeof event.input.path === 'string' ? Option.some(event.input.path) : state.path,
        }
      case 'ToolExecutionEnded': {
        switch (event.result._tag) {
          case 'Success': {
            const entries = event.result.output
            return {
              ...state,
              phase: 'completed',
              entries: [...entries],
              fileCount: entries.filter((e) => e.type === 'file').length,
              dirCount: entries.filter((e) => e.type === 'dir').length,
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
