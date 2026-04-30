import type { ToolDefinition } from "@magnitudedev/ai"
import type { Schema } from "@effect/schema"
import type { Effect } from "effect"
import type { StreamingPartial } from "./streaming-partial"

// --- ToolContext ---

export interface ToolContext<TEmission = never> {
  readonly emit: [TEmission] extends [never]
    ? never
    : (emission: TEmission) => Effect.Effect<void>
}

// --- StreamHook ---

export interface StreamHook<TInput, TEmission, TStreamState, E = never, R = never> {
  readonly initial: TStreamState
  readonly onInput: (
    input: StreamingPartial<TInput>,
    state: TStreamState,
    ctx: ToolContext<TEmission>
  ) => Effect.Effect<TStreamState, E, R>
}

// --- HarnessTool (split into erased and concrete) ---

export interface HarnessToolErased {
  readonly definition: ToolDefinition
  readonly execute: (input: unknown, ctx: unknown) => Effect.Effect<unknown, unknown, unknown>
  readonly stream?: StreamHook<unknown, unknown, unknown, unknown, unknown>
  readonly emissionSchema?: Schema.Schema<unknown, unknown, never>
  readonly errorSchema?: Schema.Schema<unknown, unknown, never>
}

export interface HarnessToolConcrete<
  TInput,
  TOutput,
  TEmission,
  TInputEncoded,
  TOutputEncoded,
  E,
  R,
> {
  readonly definition: ToolDefinition<TInput, TOutput, TInputEncoded, TOutputEncoded>
  readonly execute: (input: TInput, ctx: ToolContext<TEmission>) => Effect.Effect<TOutput, E, R>
  readonly stream?: StreamHook<TInput, TEmission, unknown, E, R>
  readonly emissionSchema?: [TEmission] extends [never] ? undefined : Schema.Schema<TEmission, unknown, never>
  readonly errorSchema?: [E] extends [never] ? undefined : Schema.Schema<E, unknown, never>
}

export type HarnessTool<
  TInput = never,
  TOutput = never,
  TEmission = never,
  TInputEncoded = TInput,
  TOutputEncoded = TOutput,
  E = never,
  R = never,
> = [TInput] extends [never]
  ? HarnessToolErased
  : HarnessToolConcrete<TInput, TOutput, TEmission, TInputEncoded, TOutputEncoded, E, R>

// Note: HarnessToolConcrete is NOT structurally assignable to HarnessToolErased due to
// function parameter contravariance (ToolContext<TEmission> vs unknown for ctx).
// The erased form is only constructed via defineHarnessTool, which handles the boundary.

// --- defineHarnessTool ---

interface DefineHarnessToolConfig<TInput, TOutput, TEmission, TInputEncoded, TOutputEncoded, E, R> {
  readonly definition: ToolDefinition<TInput, TOutput, TInputEncoded, TOutputEncoded>
  readonly execute: (input: TInput, ctx: ToolContext<TEmission>) => Effect.Effect<TOutput, E, R>
  readonly stream?: StreamHook<TInput, TEmission, unknown, E, R>
  readonly emissionSchema?: [TEmission] extends [never] ? undefined : Schema.Schema<TEmission, unknown, never>
  readonly errorSchema?: [E] extends [never] ? undefined : Schema.Schema<E, unknown, never>
}

export function defineHarnessTool<
  TInput,
  TOutput,
  TEmission = never,
  TInputEncoded = TInput,
  TOutputEncoded = TOutput,
  E = never,
  R = never,
>(
  config: DefineHarnessToolConfig<TInput, TOutput, TEmission, TInputEncoded, TOutputEncoded, E, R>
): HarnessToolConcrete<TInput, TOutput, TEmission, TInputEncoded, TOutputEncoded, E, R> {
  return {
    definition: config.definition,
    execute: config.execute,
    stream: config.stream,
    emissionSchema: config.emissionSchema,
    errorSchema: config.errorSchema,
  }
}
