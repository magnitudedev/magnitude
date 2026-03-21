import { Effect } from 'effect';

export interface ToolContext<TEmission = never> {
  emit: (value: TEmission) => Effect.Effect<void>;
}

// No-op context for tools that don't emit
export const noopToolContext: ToolContext<never> = {
  emit: () => Effect.void,
};
