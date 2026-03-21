import type { StateModel } from './state-model';

export interface ToolChain<
  TInput,
  TOutput,
  TEmission,
  TStreaming,
> {
  readonly tool: {
    inputSchema?: unknown;
    outputSchema?: unknown;
    emissionSchema?: unknown;
  };
  readonly binding: {
    _tool?: TInput;
    _streaming?: TStreaming;
  };
}

export function attachModel<
  TState,
  TInput,
  TOutput,
  TEmission,
  TStreaming,
>(
  chain: ToolChain<TInput, TOutput, TEmission, TStreaming>,
  model: StateModel<TState, TInput, TOutput, TEmission, TStreaming>,
) {
  return { ...chain, model };
}

