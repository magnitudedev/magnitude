import { defineStateModel, type BaseState } from '@magnitudedev/tools'
import { grepTool, grepXmlBinding } from '../tools/fs'

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

export const fileSearchModel = defineStateModel('fileSearch', {
  tool: grepTool,
  binding: grepXmlBinding,
})({
  initial,
  reduce: (state, event): FileSearchState => {
    switch (event.type) {
      case 'started':
        return { ...state, phase: 'streaming' }
      case 'inputUpdated':
      case 'inputReady': {
        const limitValue = event.streaming.limit?.value
        const limitStr = typeof limitValue === 'string' ? limitValue : limitValue?.toString()
        return {
          ...state,
          phase: 'streaming',
          pattern: event.streaming.pattern?.value ?? state.pattern,
          path: event.streaming.path?.value ?? state.path,
          glob: event.streaming.glob?.value ?? state.glob,
          limit: limitStr ? parseInt(limitStr, 10) || undefined : state.limit,
        }
      }
      case 'executionStarted':
      case 'emission':
      case 'awaitingApproval':
      case 'approvalGranted':
      case 'approvalRejected':
        return { ...state, phase: 'executing' }
      case 'parseError':
        return { ...state, phase: 'error', errorDetail: event.error }
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
        return { ...state, phase: 'error', errorDetail: event.error.message }
      case 'rejected':
        return { ...state, phase: 'rejected' }
      case 'interrupted':
        return { ...state, phase: 'interrupted' }
    }
  },
})
