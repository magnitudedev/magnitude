import { defineStateModel, type BaseState } from '@magnitudedev/tools'
import { createTaskTool, createTaskXmlBinding } from '../tools/task-tools'

export interface CreateTaskState extends BaseState {
  toolKey: 'createTask'
  id?: string
}

const initial: Omit<CreateTaskState, 'phase' | 'toolKey'> = {
  id: undefined,
}

export const createTaskModel = defineStateModel('createTask', {
  tool: createTaskTool,
  binding: createTaskXmlBinding,
})({
  initial,
  reduce: (state, event): CreateTaskState => {
    switch (event.type) {
      case 'started':
        return { ...state, phase: 'streaming' }
      case 'inputUpdated':
      case 'inputReady':
        return { ...state, phase: 'streaming', id: event.streaming.id?.value ?? state.id }
      case 'executionStarted':
      case 'emission':
      case 'awaitingApproval':
      case 'approvalGranted':
      case 'approvalRejected':
        return { ...state, phase: 'executing' }
      case 'parseError':
        return { ...state, phase: 'error' }
      case 'completed':
        return { ...state, phase: 'completed', id: event.output.id }
      case 'error':
        return { ...state, phase: 'error' }
      case 'rejected':
        return { ...state, phase: 'rejected' }
      case 'interrupted':
        return { ...state, phase: 'interrupted' }
    }
  },
})
