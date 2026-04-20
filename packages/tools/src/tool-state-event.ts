import type { DeepPaths } from './streaming-partial'

export type ToolResult<TOutput = unknown> =
  | { readonly _tag: 'Success'; readonly output: TOutput; readonly query: string | null }
  | { readonly _tag: 'Error'; readonly error: string }
  | { readonly _tag: 'Rejected'; readonly rejection: unknown }
  | { readonly _tag: 'Interrupted' }

export type ParseErrorDetail = {
  readonly _tag: string
  readonly detail: string
  readonly [key: string]: unknown
}

export type ToolStateEvent<TInput = unknown, TOutput = unknown, TEmission = unknown> =
  | { readonly _tag: 'ToolInputStarted' }
  | { readonly _tag: 'ToolInputFieldChunk'; readonly field: string & keyof TInput; readonly path: DeepPaths<TInput>; readonly delta: string }
  | { readonly _tag: 'ToolInputFieldComplete'; readonly field: string & keyof TInput; readonly path: DeepPaths<TInput>; readonly value: unknown }
  | { readonly _tag: 'ToolInputReady'; readonly input: TInput }
  | { readonly _tag: 'ToolInputParseError'; readonly error: ParseErrorDetail }
  | { readonly _tag: 'ToolExecutionStarted'; readonly input: TInput; readonly cached: boolean }
  | { readonly _tag: 'ToolExecutionEnded'; readonly result: ToolResult<TOutput> }
  | { readonly _tag: 'ToolEmission'; readonly value: TEmission }
