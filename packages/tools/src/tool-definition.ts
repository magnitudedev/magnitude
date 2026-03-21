import type { Schema } from '@effect/schema';
import type { Effect } from 'effect';
import type { Tool } from './tool';
import type { ToolContext } from './tool-context';

/**
 * A tool definition — pure typed function with schemas.
 * No knowledge of XML, display, or lifecycle.
 */
export interface ToolDefinition<
  TInput = unknown,
  TOutput = unknown,
  TEmission = never,
  TInputEncoded = unknown,
  TOutputEncoded = unknown,
  TEmissionEncoded = unknown,
  TError = unknown,
  TErrorEncoded = unknown,
  R = unknown,
  E = unknown
> {
  readonly name: string;
  readonly description?: string;
  readonly group?: string;
  readonly inputSchema: Schema.Schema<TInput, TInputEncoded, never>;
  readonly outputSchema: Schema.Schema<TOutput, TOutputEncoded, never>;
  readonly emissionSchema?: Schema.Schema<TEmission, TEmissionEncoded, never>;
  readonly errorSchema?: Schema.Schema<TError, TErrorEncoded, never>;
  readonly execute: (
    input: TInput,
    ctx: ToolContext<TEmission>
  ) => Effect.Effect<TOutput, E, R>;
  readonly label: (input: Partial<TInput>) => string;
}

/**
 * Structural shape for new-architecture tool definitions.
 * Kept broad to avoid variance issues with concrete generic instantiations.
 */
export interface AnyToolDefinition {
  readonly name: string;
  readonly description?: string;
  readonly group?: string;
  readonly inputSchema: Schema.Schema.Any;
  readonly outputSchema: Schema.Schema.Any;
  readonly errorSchema?: Schema.Schema.Any;
}

/**
 * Any tool contract supported by Magnitude.
 * Includes both legacy tools (`Tool.Any`) and new architecture tools (`ToolDefinition`).
 */
export type AnyTool = Tool.Any | AnyToolDefinition;

export interface ToolDefinitionConfig<
  TInput,
  TOutput,
  TEmission,
  TInputEncoded,
  TOutputEncoded,
  TEmissionEncoded,
  TError,
  TErrorEncoded,
  R,
  E
> {
  name: string;
  description?: string;
  group?: string;
  inputSchema: Schema.Schema<TInput, TInputEncoded, never>;
  outputSchema: Schema.Schema<TOutput, TOutputEncoded, never>;
  emissionSchema?: Schema.Schema<TEmission, TEmissionEncoded, never>;
  errorSchema?: Schema.Schema<TError, TErrorEncoded, never>;
  execute: (
    input: TInput,
    ctx: ToolContext<TEmission>
  ) => Effect.Effect<TOutput, E, R>;
  label: (input: Partial<TInput>) => string;
}

/**
 * Define a tool — pure typed function with schemas.
 *
 * The tool knows nothing about XML bindings, display, or lifecycle.
 * Those concerns are handled by separate contracts.
 */
export function defineTool<
  TInput,
  TOutput,
  TEmission = never,
  TInputEncoded = unknown,
  TOutputEncoded = unknown,
  TEmissionEncoded = unknown,
  TError = unknown,
  TErrorEncoded = unknown,
  R = unknown,
  E = never
>(
  config: ToolDefinitionConfig<
    TInput,
    TOutput,
    TEmission,
    TInputEncoded,
    TOutputEncoded,
    TEmissionEncoded,
    TError,
    TErrorEncoded,
    R,
    E
  >
): ToolDefinition<
  TInput,
  TOutput,
  TEmission,
  TInputEncoded,
  TOutputEncoded,
  TEmissionEncoded,
  TError,
  TErrorEncoded,
  R,
  E
> {
  return {
    name: config.name,
    description: config.description,
    group: config.group,
    inputSchema: config.inputSchema,
    outputSchema: config.outputSchema,
    emissionSchema: config.emissionSchema,
    errorSchema: config.errorSchema,
    execute: config.execute,
    label: config.label,
  };
}
