/**
 * Normalized tool state event stream consumed by state models.
 */
export type ToolStateEvent<TInput, TOutput, TEmission, TStreaming> =
  | { type: 'started' }
  | { type: 'inputUpdated'; streaming: TStreaming; changed: 'field' | 'body' | 'child'; name?: string }
  | { type: 'inputReady'; input: TInput; streaming: TStreaming }
  | { type: 'parseError'; error: string }
  | { type: 'awaitingApproval'; preview?: unknown }
  | { type: 'approvalGranted' }
  | { type: 'approvalRejected' }
  | { type: 'executionStarted' }
  | { type: 'emission'; value: TEmission }
  | { type: 'completed'; output: TOutput }
  | { type: 'error'; error: Error }
  | { type: 'rejected' }
  | { type: 'interrupted' }

/** @deprecated Use ToolStateEvent */
export type ToolLifecycleEvent<TInput, TOutput, TEmission, TStreaming> = ToolStateEvent<TInput, TOutput, TEmission, TStreaming>;
