import type { StreamingAccumulatorLike } from './state-model';

export interface ToolBinding<TInput, TEvent = unknown> {
  createAccumulator(): StreamingAccumulatorLike<TInput, TEvent>;
}
