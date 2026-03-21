import type { ToolBinding } from './tool-binding';
import type { ToolStateEvent } from './tool-state-event';

export type Phase = 'streaming' | 'executing' | 'completed' | 'error' | 'rejected' | 'interrupted';

export type BaseState = { phase: Phase };

export interface StateModel<TState, TInput, TOutput, TEmission, TStreaming> {
  readonly initial: TState;
  readonly reduce: (
    state: TState,
    event: ToolStateEvent<TInput, TOutput, TEmission, TStreaming>
  ) => TState;
}

export function defineStateModel<TInput, TOutput, TEmission, TStreaming>(
  chain: {
    tool: { inputSchema: { Type: TInput }; outputSchema: { Type: TOutput }; emissionSchema?: { Type: TEmission } };
    binding: ToolBinding<TInput, TStreaming>;
  }
) {
  void chain;

  return <TExtra extends Record<string, unknown>>(
    config: {
      initial: TExtra;
      reduce: (
        state: BaseState & TExtra,
        event: ToolStateEvent<TInput, TOutput, TEmission, TStreaming>
      ) => BaseState & TExtra;
    }
  ): StateModel<BaseState & TExtra, TInput, TOutput, TEmission, TStreaming> => {
    const initial = { ...({ phase: 'streaming' } satisfies BaseState), ...config.initial };
    return { initial, reduce: config.reduce };
  };
}
