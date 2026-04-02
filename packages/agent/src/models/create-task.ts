import { defineStateModel, type BaseState } from '@magnitudedev/tools'
import { createTaskTool, createTaskXmlBinding } from '../tools/task-tools'

export interface CreateTaskState extends BaseState {
  toolKey: 'createTask'
  taskId?: string
  type?: string
  title?: string
}

const initial: Omit<CreateTaskState, 'phase' | 'toolKey'> = {
  taskId: undefined,
  type: undefined,
  title: undefined,
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
        return {
          ...state,
          phase: 'streaming',
          taskId: event.streaming.taskId?.value ?? state.taskId,
          type: event.streaming.type?.value ?? state.type,
          title: event.streaming.title?.value ?? state.title,
        }
      case 'executionStarted':
      case 'emission':
      case 'awaitingApproval':
      case 'approvalGranted':
      case 'approvalRejected':
        return { ...state, phase: 'executing' }
      case 'parseError':
        return { ...state, phase: 'error' }
      case 'completed':
        return {
          ...state,
          phase: 'completed',
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
