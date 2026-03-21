import { defineStateModel, type BaseState, type EditDiff } from '@magnitudedev/tools'
import { editTool, editXmlBinding } from '../tools/fs'

export interface FileEditState extends BaseState {
  path?: string
  oldText: string
  newText: string
  diffs: EditDiff[]
}

const initial: Omit<FileEditState, 'phase'> = {
  path: undefined,
  oldText: '',
  newText: '',
  diffs: [],
}

export const fileEditModel = defineStateModel({
  tool: editTool,
  binding: editXmlBinding,
})({
  initial,
  reduce: (state, event): FileEditState => {
    switch (event.type) {
      case 'started':
        return { ...state, phase: 'streaming' }
      case 'inputUpdated':
      case 'inputReady':
        return {
          ...state,
          phase: 'streaming',
          path: event.streaming.fields.path ?? state.path,
          oldText: event.streaming.children?.old?.[0]?.body ?? state.oldText,
          newText: event.streaming.children?.new?.[0]?.body ?? state.newText,
        }
      case 'executionStarted':
      case 'awaitingApproval':
      case 'approvalGranted':
      case 'approvalRejected':
      case 'parseError':
        return { ...state, phase: 'executing' }
      case 'emission':
        return event.value.type === 'edit_diff'
          ? { ...state, phase: 'executing', path: event.value.path, diffs: [...event.value.diffs] }
          : state
      case 'completed':
        return { ...state, phase: 'completed' }
      case 'error':
        return { ...state, phase: 'error' }
      case 'rejected':
        return { ...state, phase: 'rejected' }
      case 'interrupted':
        return { ...state, phase: 'interrupted' }
    }
  },
})
