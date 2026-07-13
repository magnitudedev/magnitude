import type { JsonEncodedSchema, JsonObjectEncodedSchema } from "@magnitudedev/utils/schema"
import type { Schema } from "effect"

/**
 * Erased form — for acceptance in collections and function signatures.
 */
export interface ToolDefinitionErased {
  readonly name: string
  readonly description: string
  readonly inputSchema: Schema.Schema.AnyNoContext
  readonly outputSchema: Schema.Schema.AnyNoContext
}

/**
 * Concrete form — full type safety for input/output schemas.
 */
export interface ToolDefinitionConcrete<
  TInputSchema extends Schema.Schema.AnyNoContext,
  TOutputSchema extends Schema.Schema.AnyNoContext,
> {
  readonly name: string
  readonly description: string
  readonly inputSchema: JsonObjectEncodedSchema<TInputSchema>
  readonly outputSchema: JsonEncodedSchema<TOutputSchema>
}

/**
 * Never-switched: bare `ToolDefinition` resolves to erased form,
 * `ToolDefinition<I, O>` resolves to concrete form.
 */
export type ToolDefinition<
  TInputSchema extends Schema.Schema.AnyNoContext = never,
  TOutputSchema extends Schema.Schema.AnyNoContext = never,
> = [TInputSchema] extends [never]
  ? ToolDefinitionErased
  : ToolDefinitionConcrete<TInputSchema, TOutputSchema>

export function defineTool<
  TInputSchema extends Schema.Schema.AnyNoContext,
  TOutputSchema extends Schema.Schema.AnyNoContext,
>(
  definition: ToolDefinitionConcrete<TInputSchema, TOutputSchema>,
): ToolDefinitionConcrete<TInputSchema, TOutputSchema> {
  return definition
}
