import type { StateModel } from './state-model';
import type { ToolStateEvent } from './tool-state-event';

export type CallState<TState, TStreaming> = {
  state: TState;
  streaming: TStreaming;
};

// What every display renderer receives — state + metadata + optional extra props
export type DisplayProps<TState, TOutput, TExtra = {}> = {
  state: TState;
  label: string;
  result?: ToolResult<TOutput>;
  isExpanded: boolean;
  onToggle: () => void;
} & TExtra;

// ToolResult — the app-level result of a tool execution
export interface ToolResult<TOutput> {
  status: 'success' | 'error' | 'rejected' | 'interrupted';
  output?: TOutput;
  message?: string;
  reason?: string;
}

// Display renderer interface — no streaming, just state
export interface DisplayConfig<TState, TRender, TOutput, TExtra = {}> {
  render: (props: DisplayProps<TState, TOutput, TExtra>) => TRender;
  summary: (state: TState) => string;
}

export interface Display<TState, TRender, TOutput, TExtra = {}> {
  readonly render: (props: DisplayProps<TState, TOutput, TExtra>) => TRender;
  readonly summary: (state: TState) => string;
}

export function defineDisplay<TState, TInput = unknown, TOutput = unknown, TEmission = never, TStreaming = unknown, TRender = unknown, TExtra = {}>(
  _model: StateModel<TState, TInput, TOutput, TEmission, TStreaming>,
  config: DisplayConfig<TState, TRender, TOutput, TExtra>
): Display<TState, TRender, TOutput, TExtra> {
  return {
    render: config.render,
    summary: config.summary,
  };
}

/**
 * Tool display binding.
 * Defaults exist only to support erased internal heterogeneous storage.
 */
export interface ToolDisplayBinding<
  TState = object,
  TStreaming = unknown,
  TInput = unknown,
  TOutput = unknown,
  TEmission = unknown,
  TRender = unknown
> {
  createCallState(): CallState<TState, TStreaming>;
  handleStarted(cs: CallState<TState, TStreaming>): void;
  handleInputUpdated(cs: CallState<TState, TStreaming>, streaming: TStreaming, changed: 'field' | 'body' | 'child', name?: string): void;
  handleInputReady(cs: CallState<TState, TStreaming>, input: TInput, streaming: TStreaming): void;
  handleExecutionStarted(cs: CallState<TState, TStreaming>): void;
  handleEmission(cs: CallState<TState, TStreaming>, value: TEmission): void;
  handleCompleted(cs: CallState<TState, TStreaming>, output: TOutput): void;
  handleError(cs: CallState<TState, TStreaming>, error: Error): void;
  handleRejected(cs: CallState<TState, TStreaming>): void;
  handleInterrupted(cs: CallState<TState, TStreaming>): void;
  handleAwaitingApproval(cs: CallState<TState, TStreaming>, preview?: unknown): void;
  handleApprovalGranted(cs: CallState<TState, TStreaming>): void;
  handleApprovalRejected(cs: CallState<TState, TStreaming>): void;
  render(cs: CallState<TState, TStreaming>, props: { label: string; result?: ToolResult<TOutput>; isExpanded: boolean; onToggle: () => void } & Record<string, unknown>): TRender;
  summary(cs: CallState<TState, TStreaming>): string;
}

export function createBinding<TState extends object, TInput, TOutput, TEmission, TStreaming, TRender>(
  model: StateModel<TState, TInput, TOutput, TEmission, TStreaming>,
  display: Display<TState, TRender, TOutput>,
  initialStreaming: TStreaming,
): ToolDisplayBinding<TState, TStreaming, TInput, TOutput, TEmission, TRender> {
  const reduce = (
    cs: CallState<TState, TStreaming>,
    event: ToolStateEvent<TInput, TOutput, TEmission, TStreaming>
  ) => {
    cs.state = model.reduce(cs.state, event);
  };

  return {
    createCallState: () => ({ state: model.initial, streaming: initialStreaming }),
    handleStarted: (cs) => reduce(cs, { type: 'started' }),
    handleInputUpdated: (cs, streaming, changed, name) => {
      cs.streaming = streaming;
      reduce(cs, { type: 'inputUpdated', streaming, changed, name });
    },
    handleInputReady: (cs, input, streaming) => {
      cs.streaming = streaming;
      reduce(cs, { type: 'inputReady', input, streaming });
    },
    handleExecutionStarted: (cs) => reduce(cs, { type: 'executionStarted' }),
    handleEmission: (cs, value) => reduce(cs, { type: 'emission', value }),
    handleCompleted: (cs, output) => reduce(cs, { type: 'completed', output }),
    handleError: (cs, error) => reduce(cs, { type: 'error', error }),
    handleRejected: (cs) => reduce(cs, { type: 'rejected' }),
    handleInterrupted: (cs) => reduce(cs, { type: 'interrupted' }),
    handleAwaitingApproval: (cs, preview) => reduce(cs, { type: 'awaitingApproval', preview }),
    handleApprovalGranted: (cs) => reduce(cs, { type: 'approvalGranted' }),
    handleApprovalRejected: (cs) => reduce(cs, { type: 'approvalRejected' }),
    render: (cs, props) => {
      const { label, result, isExpanded, onToggle, ...extra } = props;
      return display.render({
        state: cs.state,
        label,
        result,
        isExpanded,
        onToggle,
        ...extra,
      });
    },
    summary: (cs) => display.summary(cs.state),
  };
}

export type BindingState<B> = B extends ToolDisplayBinding<infer S, any, any, any, any, any> ? S : never;
export type BindingStreaming<B> = B extends ToolDisplayBinding<any, infer S, any, any, any, any> ? S : never;
export type BindingInput<B> = B extends ToolDisplayBinding<any, any, infer I, any, any, any> ? I : never;
export type BindingOutput<B> = B extends ToolDisplayBinding<any, any, any, infer O, any, any> ? O : never;
export type BindingEmission<B> = B extends ToolDisplayBinding<any, any, any, any, infer E, any> ? E : never;
