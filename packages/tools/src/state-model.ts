import type { ToolStateEvent } from './tool-state-event';

export type Phase = 'streaming' | 'executing' | 'completed' | 'error' | 'rejected' | 'interrupted';

export type BaseState = { phase: Phase };

/** Minimal streaming accumulator contract for state models */
export interface StreamingAccumulatorLike<TInput> {
  ingest(event: unknown): void;
  readonly current: import('./streaming-partial').StreamingPartial<TInput>;
  reset(): void;
}

export interface StateModel<TState, TInput, TOutput, TEmission> {
  readonly initial: TState;
  readonly reduce: (
    state: TState,
    event: ToolStateEvent<TInput, TOutput, TEmission>
  ) => TState;
  readonly binding: { createAccumulator(): StreamingAccumulatorLike<TInput> };
}

export function defineStateModel<TToolKey extends string, TInput, TOutput, TEmission>(
  toolKey: TToolKey,
  chain: {
    tool: { inputSchema: { Type: TInput }; outputSchema: { Type: TOutput }; emissionSchema?: { Type: TEmission } };
    binding: { readonly _tool?: TInput };
  }
) {
  void chain;

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
    return { initial, reduce: config.reduce, binding: chain.binding as any };
  };
}
