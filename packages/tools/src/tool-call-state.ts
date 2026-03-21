import type { StateModel } from './state-model';
import type { ToolStateEvent } from './tool-state-event';

export class ToolCallState<TState, TStreaming> {
  state: TState;
  streaming: TStreaming;

  constructor(
    private readonly model: StateModel<TState, any, any, any, TStreaming>,
    initialStreaming: TStreaming,
  ) {
    this.state = model.initial;
    this.streaming = initialStreaming;
  }

  dispatch(event: ToolStateEvent<any, any, any, TStreaming>): void {
    this.state = this.model.reduce(this.state, event);
    if (event.type === 'inputUpdated' || event.type === 'inputReady') {
      this.streaming = event.streaming;
    }
  }

  snapshot(): { state: TState; streaming: TStreaming } {
    return { state: this.state, streaming: this.streaming };
  }
}

export function createToolCallState<TState, TStreaming>(
  model: StateModel<TState, any, any, any, TStreaming>,
  initialStreaming: TStreaming,
): ToolCallState<TState, TStreaming> {
  return new ToolCallState(model, initialStreaming);
}
