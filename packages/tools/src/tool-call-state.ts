import type { StateModel } from './state-model';
import type { ToolStateEvent } from './tool-state-event';

export class ToolCallState<TState> {
  state: TState;

  constructor(
    private readonly model: StateModel<TState, any, any, any>,
  ) {
    this.state = model.initial;
  }

  dispatch(event: ToolStateEvent<any, any, any>): void {
    this.state = this.model.reduce(this.state, event);
  }

  snapshot(): { state: TState } {
    return { state: this.state };
  }
}

export function createToolCallState<TState>(
  model: StateModel<TState, any, any, any>,
): ToolCallState<TState> {
  return new ToolCallState(model);
}
