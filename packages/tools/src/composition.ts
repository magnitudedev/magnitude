import type { StateModel } from './state-model';

export interface ToolChain<
  TInput,
  TOutput,
  TEmission,
> {
  readonly tool: {
    inputSchema?: unknown;
    outputSchema?: unknown;
    emissionSchema?: unknown;
  };
  readonly binding: {
    _tool?: TInput;
  };
}

export function attachModel<
  TState,
  TInput,
  TOutput,
  TEmission,
>(
  chain: ToolChain<TInput, TOutput, TEmission>,
  model: StateModel<TState, TInput, TOutput, TEmission>,
) {
  return { ...chain, model };
}

