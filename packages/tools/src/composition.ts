import type { StateModel } from './state-model';
import type { Display, ToolDisplayBinding } from './display';
import { createBinding } from './display';

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

export function attachDisplay<
  TState extends object,
  TInput,
  TOutput,
  TEmission,
  TStreaming,
  TRender,
>(
  composed: {
    tool: ToolChain<TInput, TOutput, TEmission, TStreaming>['tool'];
    binding: ToolChain<TInput, TOutput, TEmission, TStreaming>['binding'];
    model: StateModel<TState, TInput, TOutput, TEmission, TStreaming>;
  },
  display: Display<TState, TRender, TOutput>,
  initialStreaming: TStreaming,
): {
  tool: ToolChain<TInput, TOutput, TEmission, TStreaming>['tool'];
  binding: ToolChain<TInput, TOutput, TEmission, TStreaming>['binding'];
  model: StateModel<TState, TInput, TOutput, TEmission, TStreaming>;
  display: Display<TState, TRender, TOutput>;
  displayBinding: ToolDisplayBinding<TState, TStreaming, TInput, TOutput, TEmission, TRender>;
} {
  const displayBinding = createBinding(composed.model, display, initialStreaming);
  return { ...composed, display, displayBinding };
}
