import type { StreamingPartial } from './streaming-partial';
import type { ToolBinding } from './tool-binding';
import type { ToolStateEvent } from './tool-state-event';

export type Phase = 'streaming' | 'executing' | 'completed' | 'error' | 'rejected' | 'interrupted';

export type BaseState = { phase: Phase };

/** Minimal streaming accumulator contract for state models */
export interface StreamingAccumulatorLike<TInput, TEvent = unknown> {
  ingest(event: TEvent): void;
  readonly current: StreamingPartial<TInput>;
  reset(): void;
}

export interface StateModel<TState, TInput, TOutput, TEmission, TEvent = unknown> {
  readonly initial: TState;
  readonly reduce: (
    state: TState,
    event: ToolStateEvent<TInput, TOutput, TEmission>
  ) => TState;
  readonly binding: { createAccumulator(): StreamingAccumulatorLike<TInput, TEvent> };
}

export function defineStateModel<TToolKey extends string, TInput, TOutput, TEmission, TEvent = unknown>(
  toolKey: TToolKey,
  chain: {
    tool: { inputSchema: { Type: TInput }; outputSchema: { Type: TOutput }; emissionSchema?: { Type: TEmission } };
    binding: ToolBinding<TInput, TEvent>;
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
  ): StateModel<{ toolKey: TToolKey } & BaseState & TExtra, TInput, TOutput, TEmission, TEvent> => {
    const initial = { toolKey, ...({ phase: 'streaming' } satisfies BaseState), ...config.initial };
    return { initial, reduce: config.reduce, binding: chain.binding };
  };
}
