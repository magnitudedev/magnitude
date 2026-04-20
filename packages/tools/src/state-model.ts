import type { ToolStateEvent } from './tool-state-event'

export type Phase = 'streaming' | 'executing' | 'completed' | 'error' | 'rejected' | 'interrupted';

export type BaseState = { phase: Phase };

export interface StateModel<TState, TInput, TOutput, TEmission> {
  readonly initial: TState;
  readonly reduce: (
    state: TState,
    event: ToolStateEvent<TInput, TOutput, TEmission>
  ) => TState;
}

export function defineStateModel<TToolKey extends string, TInput, TOutput, TEmission>(
  toolKey: TToolKey,
  tool: { inputSchema: { Type: TInput }; outputSchema: { Type: TOutput }; emissionSchema?: { Type: TEmission } },
) {
  return <TExtra extends Record<string, unknown>>(
    config: {
      initial: TExtra;
      reduce: (
        state: { toolKey: TToolKey } & BaseState & TExtra,
        event: ToolStateEvent<TInput, TOutput, TEmission>
      ) => { toolKey: TToolKey } & BaseState & TExtra;
    }
  ): StateModel<{ toolKey: TToolKey } & BaseState & TExtra, TInput, TOutput, TEmission> => {
    const initial = { toolKey, ...({ phase: 'streaming' } satisfies BaseState), ...config.initial };
    return { initial, reduce: config.reduce };
  };
}
