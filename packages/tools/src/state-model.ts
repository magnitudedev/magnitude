import type { StreamingPartial } from './streaming-partial';
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
  readonly createAccumulator: () => StreamingAccumulatorLike<TInput, TEvent>;
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
  ): StateModel<{ toolKey: TToolKey } & BaseState & TExtra, TInput, TOutput, TEmission, unknown> => {
    const initial = { toolKey, ...({ phase: 'streaming' } satisfies BaseState), ...config.initial };
    // createAccumulator is no longer provided by the state model — it's created externally
    // from the tool's schema in tool-handle.ts
    return { 
      initial, 
      reduce: config.reduce, 
      createAccumulator: () => {
        throw new Error('createAccumulator should not be called directly — use createParameterAccumulator from tool schema')
      } 
    };
  };
}
