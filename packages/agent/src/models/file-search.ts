import { defineStateModel, type BaseState } from '@magnitudedev/tools'
import { searchTool, searchXmlBinding } from '../tools/fs'

type SearchMatch = { file: string; match: string }

export interface FileSearchState extends BaseState {
  pattern?: string
  path?: string
  glob?: string
  matches: SearchMatch[]
  matchCount: number
  fileCount: number
}

const initial: Omit<FileSearchState, 'phase'> = {
  pattern: undefined,
  path: undefined,
  glob: undefined,
  matches: [],
  matchCount: 0,
  fileCount: 0,
}

export const fileSearchModel = defineStateModel({
  tool: searchTool,
  binding: searchXmlBinding,
})({
  initial,
  reduce: (state, event): FileSearchState => {
    switch (event.type) {
      case 'started':
        return { ...state, phase: 'streaming' }
      case 'inputUpdated':
      case 'inputReady':
        return {
          ...state,
          phase: 'streaming',
          pattern: event.streaming.fields.pattern ?? state.pattern,
          path: event.streaming.fields.path ?? state.path,
          glob: event.streaming.fields.glob ?? state.glob,
        }
      case 'executionStarted':
      case 'emission':
      case 'awaitingApproval':
      case 'approvalGranted':
      case 'approvalRejected':
      case 'parseError':
        return { ...state, phase: 'executing' }
      case 'completed': {
        const matches = [...event.output]
        return {
          ...state,
          phase: 'completed',
          matches,
          matchCount: matches.length,
          fileCount: new Set(matches.map((m) => m.file)).size,
        }
      }
      case 'error':
        return { ...state, phase: 'error' }
      case 'rejected':
        return { ...state, phase: 'rejected' }
      case 'interrupted':
        return { ...state, phase: 'interrupted' }
    }
  },
})
