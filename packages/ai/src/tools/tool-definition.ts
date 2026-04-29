import type { Schema } from "effect"

/**
 * Erased form — for acceptance in collections and function signatures.
 */
export interface ToolDefinitionErased {
  readonly name: string
  readonly description: string
  readonly inputSchema: Schema.Schema.Any
  readonly outputSchema: Schema.Schema.Any
}

/**
 * Concrete form — full type safety for input/output schemas.
 */
export interface ToolDefinitionConcrete<
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

/**
 * Never-switched: bare `ToolDefinition` resolves to erased form,
 * `ToolDefinition<I, O>` resolves to concrete form.
 */
export type ToolDefinition<
  TInput = never,
  TOutput = never,
  TInputEncoded = TInput,
  TOutputEncoded = TOutput,
> = [TInput] extends [never]
  ? ToolDefinitionErased
  : ToolDefinitionConcrete<TInput, TOutput, TInputEncoded, TOutputEncoded>

export function defineTool<
  TInput,
  TOutput,
  TInputEncoded = TInput,
  TOutputEncoded = TOutput,
>(
  definition: ToolDefinitionConcrete<TInput, TOutput, TInputEncoded, TOutputEncoded>,
): ToolDefinitionConcrete<TInput, TOutput, TInputEncoded, TOutputEncoded> {
  return definition
}
