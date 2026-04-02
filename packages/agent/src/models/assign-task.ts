import { defineStateModel, type BaseState } from '@magnitudedev/tools'
import { assignTaskTool, assignTaskXmlBinding } from '../tools/task-tools'

export interface AssignTaskState extends BaseState {
  toolKey: 'assignTask'
  taskId?: string
  assignee?: string
  agentId?: string
  forkId?: string
  message?: string
}

const initial: Omit<AssignTaskState, 'phase' | 'toolKey'> = {
  taskId: undefined,
  assignee: undefined,
  agentId: undefined,
  forkId: undefined,
  message: undefined,
}

export const assignTaskModel = defineStateModel('assignTask', {
  tool: assignTaskTool,
  binding: assignTaskXmlBinding,
})({
  initial,
  reduce: (state, event): AssignTaskState => {
    switch (event.type) {
      case 'started':
        return { ...state, phase: 'streaming' }
      case 'inputUpdated':
      case 'inputReady':
        return {
          ...state,
          phase: 'streaming',
          taskId: event.streaming.taskId?.value ?? state.taskId,
          assignee: event.streaming.assignee?.value ?? state.assignee,
          message: event.streaming.message?.value ?? state.message,
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
          agentId: event.output.agentId ?? state.agentId,
          forkId: event.output.forkId ?? state.forkId,
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
