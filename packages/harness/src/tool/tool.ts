import type { JsonEncodedSchema } from "@magnitudedev/utils/schema"
import type { ToolDefinition, ValidationIssue } from "@magnitudedev/ai"
import type { Schema } from "effect"
import { Data, type Effect } from "effect"
import type { StreamingPartial } from "@magnitudedev/ai"

// --- StreamValidationError ---

export class StreamValidationError extends Data.TaggedError("StreamValidationError")<{
  readonly message: string
}> {}

// --- ToolContext ---

export interface ToolContext<TEmission = never> {
  readonly emit: [TEmission] extends [never]
    ? never
    : (emission: TEmission) => Effect.Effect<void>
}

// --- StreamHook ---

export interface StreamHook<TInput, TEmission, TStreamState, RStream = never> {
  readonly initial: TStreamState
  readonly onInput: (
    input: StreamingPartial<TInput>,
    state: TStreamState,
    ctx: ToolContext<TEmission>
  ) => Effect.Effect<TStreamState, StreamValidationError, RStream>
}

// --- HarnessTool (split into erased and concrete) ---

export interface HarnessToolErased {
  readonly definition: ToolDefinition
  readonly execute: (input: any, ctx: any) => Effect.Effect<any, any, any>
  readonly stream?: StreamHook<any, any, any, any>
  readonly emissionSchema?: Schema.Schema.AnyNoContext | undefined
  readonly errorSchema?: Schema.Schema.AnyNoContext | undefined
}

export interface HarnessToolConcrete<
  TInputSchema extends Schema.Schema.AnyNoContext,
  TOutputSchema extends Schema.Schema.AnyNoContext,
  TEmissionSchema extends Schema.Schema.AnyNoContext | undefined,
  TErrorSchema extends Schema.Schema.AnyNoContext | undefined,
  RExecute,
  RStream = never,
  TStreamState = unknown,
> {
  readonly definition: ToolDefinition<TInputSchema, TOutputSchema>
  readonly execute: (
    input: Schema.Schema.Type<TInputSchema>,
    ctx: ToolContext<SchemaTypeOrNever<TEmissionSchema>>,
  ) => Effect.Effect<Schema.Schema.Type<TOutputSchema>, SchemaTypeOrNever<TErrorSchema>, RExecute>
  readonly stream?: StreamHook<Schema.Schema.Type<TInputSchema>, SchemaTypeOrNever<TEmissionSchema>, TStreamState, RStream>
  readonly emissionSchema?: SchemaOrUndefined<TEmissionSchema>
  readonly errorSchema?: SchemaOrUndefined<TErrorSchema>
}

export type HarnessTool<
  TInputSchema extends Schema.Schema.AnyNoContext = never,
  TOutputSchema extends Schema.Schema.AnyNoContext = never,
  TEmissionSchema extends Schema.Schema.AnyNoContext | undefined = undefined,
  TErrorSchema extends Schema.Schema.AnyNoContext | undefined = undefined,
  RExecute = never,
  RStream = never,
> = [TInputSchema] extends [never]
  ? HarnessToolErased
  : HarnessToolConcrete<TInputSchema, TOutputSchema, TEmissionSchema, TErrorSchema, RExecute, RStream>

// Note: HarnessToolConcrete is NOT structurally assignable to HarnessToolErased due to
// function parameter contravariance (ToolContext<TEmission> vs unknown for ctx).
// The erased form is only constructed via defineHarnessTool, which handles the boundary.

// --- defineHarnessTool ---

type SchemaTypeOrNever<TSchema extends Schema.Schema.AnyNoContext | undefined> =
  TSchema extends Schema.Schema.AnyNoContext ? Schema.Schema.Type<TSchema> : never

type SchemaOrUndefined<TSchema extends Schema.Schema.AnyNoContext | undefined> =
  TSchema extends Schema.Schema.AnyNoContext ? JsonEncodedSchema<TSchema> : undefined

interface DefineHarnessToolConfig<
  TInputSchema extends Schema.Schema.AnyNoContext,
  TOutputSchema extends Schema.Schema.AnyNoContext,
  TEmissionSchema extends Schema.Schema.AnyNoContext | undefined,
  TErrorSchema extends Schema.Schema.AnyNoContext | undefined,
  RExecute,
  RStream,
  TStreamState = unknown,
> {
  readonly definition: ToolDefinition<TInputSchema, TOutputSchema>
  readonly execute: (
    input: Schema.Schema.Type<TInputSchema>,
    ctx: ToolContext<SchemaTypeOrNever<TEmissionSchema>>,
  ) => Effect.Effect<Schema.Schema.Type<TOutputSchema>, SchemaTypeOrNever<TErrorSchema>, RExecute>
  readonly stream?: StreamHook<Schema.Schema.Type<TInputSchema>, SchemaTypeOrNever<TEmissionSchema>, TStreamState, RStream>
  readonly emissionSchema?: SchemaOrUndefined<TEmissionSchema>
  readonly errorSchema?: SchemaOrUndefined<TErrorSchema>
}

export function defineHarnessTool<
  TInputSchema extends Schema.Schema.AnyNoContext,
  TOutputSchema extends Schema.Schema.AnyNoContext,
  TEmissionSchema extends Schema.Schema.AnyNoContext | undefined = undefined,
  TErrorSchema extends Schema.Schema.AnyNoContext | undefined = undefined,
  RExecute = never,
  RStream = never,
  TStreamState = unknown,
>(
  config: DefineHarnessToolConfig<TInputSchema, TOutputSchema, TEmissionSchema, TErrorSchema, RExecute, RStream, TStreamState>
): HarnessToolConcrete<TInputSchema, TOutputSchema, TEmissionSchema, TErrorSchema, RExecute, RStream, TStreamState> {
  return {
    definition: config.definition,
    execute: config.execute,
    stream: config.stream,
    emissionSchema: config.emissionSchema,
    errorSchema: config.errorSchema,
  }
}
