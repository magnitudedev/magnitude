import type { StateModel } from '../state-model';
import type { ToolStateEvent } from '../tool-state-event';

export type { Phase, BaseState } from '../state-model';
export type { Phase as PhaseType } from '../state-model';

export type DefaultState = { phase: import('../state-model').Phase };

export const defaultModel: StateModel<DefaultState, unknown, unknown, unknown, unknown> = {
  initial: { phase: 'streaming' },
  reduce: (state: DefaultState, event: ToolStateEvent<unknown, unknown, unknown, unknown>): DefaultState => {
    switch (event.type) {
      case 'started':
        return { phase: 'streaming' };
      case 'executionStarted':
        return { phase: 'executing' };
      case 'completed':
        return { phase: 'completed' };
      case 'error':
        return { phase: 'error' };
      case 'rejected':
      case 'approvalRejected':
        return { phase: 'rejected' };
      case 'interrupted':
        return { phase: 'interrupted' };
      default:
        return state;
    }
  },
};
