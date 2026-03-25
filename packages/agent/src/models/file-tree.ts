import { defineStateModel, type BaseState } from '@magnitudedev/tools'
import { treeTool, treeXmlBinding } from '../tools/fs'

type TreeEntry = { path: string; name: string; type: 'file' | 'dir'; depth: number }

export interface FileTreeState extends BaseState {
  toolKey: 'fileTree'
  path?: string
  entries: TreeEntry[]
  fileCount: number
  dirCount: number
  errorDetail?: string
}

const initial: Omit<FileTreeState, 'phase' | 'toolKey'> = {
  path: undefined,
  entries: [],
  fileCount: 0,
  dirCount: 0,
  errorDetail: undefined,
}

export const fileTreeModel = defineStateModel('fileTree', {
  tool: treeTool,
  binding: treeXmlBinding,
})({
  initial,
  reduce: (state, event): FileTreeState => {
    switch (event.type) {
      case 'started':
        return { ...state, phase: 'streaming' }
      case 'inputUpdated':
      case 'inputReady':
        return { ...state, phase: 'streaming', path: event.streaming.path?.value ?? state.path }
      case 'executionStarted':
      case 'emission':
      case 'awaitingApproval':
      case 'approvalGranted':
      case 'approvalRejected':
      case 'parseError':
        return { ...state, phase: 'executing' }
      case 'completed': {
        const entries = event.output as TreeEntry[]
        return {
          ...state,
          phase: 'completed',
          entries: [...entries],
          fileCount: entries.filter((entry) => entry.type === 'file').length,
          dirCount: entries.filter((entry) => entry.type === 'dir').length,
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
