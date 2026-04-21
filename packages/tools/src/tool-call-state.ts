import type { StateModel } from "./state-model";
import type { ToolStateEvent } from "./tool-state-event";

export class ToolCallState<TState, TInput = unknown, TOutput = unknown, TEmission = unknown> {
  state: TState;

  constructor(
    private readonly model: StateModel<TState, TInput, TOutput, TEmission>,
  ) {
    this.state = model.initial;
  }

  dispatch(event: ToolStateEvent<TInput, TOutput, TEmission>): void {
    this.state = this.model.reduce(this.state, event);
  }

  snapshot(): { state: TState } {
    return { state: this.state };
  }
}

export function createToolCallState<TState, TInput = unknown, TOutput = unknown, TEmission = unknown>(
  model: StateModel<TState, TInput, TOutput, TEmission>,
): ToolCallState<TState, TInput, TOutput, TEmission> {
  return new ToolCallState(model);
}
