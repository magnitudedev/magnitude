import type { StateModel } from './state-model';
import type { ToolBinding } from './tool-binding';

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
  readonly binding: ToolBinding<TInput>;
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

