import type { Schema } from "effect"

export interface ToolDefinition<
  TInput,
  TOutput,
  TInputEncoded = TInput,
  TOutputEncoded = TOutput,
> {
  readonly name: string
  readonly description: string
  readonly inputSchema: Schema.Schema<TInput, TInputEncoded, never>
  readonly outputSchema: Schema.Schema<TOutput, TOutputEncoded, never>
}

export function defineTool<
  TInput,
  TOutput,
  TInputEncoded = TInput,
  TOutputEncoded = TOutput,
>(
  definition: ToolDefinition<TInput, TOutput, TInputEncoded, TOutputEncoded>,
): ToolDefinition<TInput, TOutput, TInputEncoded, TOutputEncoded> {
  return definition
}
