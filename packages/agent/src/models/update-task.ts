import { defineStateModel, type BaseState } from '@magnitudedev/tools'
import { updateTaskTool, updateTaskXmlBinding } from '../tools/task-tools'

export interface UpdateTaskState extends BaseState {
  toolKey: 'updateTask'
  taskId?: string
  parent?: string
  status?: string
  title?: string
}

const initial: Omit<UpdateTaskState, 'phase' | 'toolKey'> = {
  taskId: undefined,
  parent: undefined,
  status: undefined,
  title: undefined,
}

export const updateTaskModel = defineStateModel('updateTask', {
  tool: updateTaskTool,
  binding: updateTaskXmlBinding,
})({
  initial,
  reduce: (state, event): UpdateTaskState => {
    switch (event.type) {
      case 'started':
        return { ...state, phase: 'streaming' }
      case 'inputUpdated':
      case 'inputReady':
        return {
          ...state,
          phase: 'streaming',
          taskId: event.streaming.taskId?.value ?? state.taskId,
          parent: event.streaming.parent?.value ?? state.parent,
          status: event.streaming.status?.value ?? state.status,
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
          taskId: event.output.taskId,
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
