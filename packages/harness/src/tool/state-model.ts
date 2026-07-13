import type { ToolError, ToolLifecycleEvent } from "../events"
import { Option, type Effect, Schema } from "effect"

// ── Phase ────────────────────────────────────────────────────────────

export const PhaseSchema = Schema.Literal("streaming", "executing", "completed", "error", "rejected", "interrupted")
export type Phase = typeof PhaseSchema.Type

// ── Base State ───────────────────────────────────────────────────────

export const BaseStateSchema = Schema.Struct({
  phase: PhaseSchema,
  errorMessage: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
})
export type BaseState = typeof BaseStateSchema.Type

// ── State Model ──────────────────────────────────────────────────────

interface StateModelErased<TStateSchema extends Schema.Schema.AnyNoContext = typeof BaseStateSchema> {
  readonly stateSchema: TStateSchema
  readonly initial: Schema.Schema.Type<TStateSchema>
  readonly reduce: (state: Schema.Schema.Type<TStateSchema>, event: ToolLifecycleEvent) => Schema.Schema.Type<TStateSchema>
}

interface StateModelConcrete<
  TStateSchema extends Schema.Schema.AnyNoContext,
  TInput,
  TOutput,
  TEmission,
  TError extends ToolError,
> {
  readonly stateSchema: TStateSchema
  readonly initial: Schema.Schema.Type<TStateSchema>
  readonly reduce: (state: Schema.Schema.Type<TStateSchema>, event: ToolLifecycleEvent<TInput, TOutput, TEmission, TError>) => Schema.Schema.Type<TStateSchema>
}

export type StateModel<
  TStateSchema extends Schema.Schema.AnyNoContext = Schema.Schema.AnyNoContext,
  TInput = never,
  TOutput = never,
  TEmission = never,
  TError extends ToolError = never,
> = StateModelConcrete<TStateSchema, TInput, TOutput, TEmission, TError>

// ── Inference-only tool shape ────────────────────────────────────────

/**
 * Minimal shape used purely for generic type inference in defineStateModel.
 * Avoids coupling to the full HarnessTool type.
 */
interface ToolTypeCarrier<TInput, TOutput, TEmission, TError extends ToolError = never> {
  readonly execute: (input: TInput, ctx: any) => Effect.Effect<TOutput, any, any>
  readonly emissionSchema?: { readonly Type: TEmission }
  readonly errorSchema?: { readonly Type: TError }
}

interface ToolSchemaCarrier<
  TInputSchema extends Schema.Schema.AnyNoContext,
  TOutputSchema extends Schema.Schema.AnyNoContext,
  TEmissionSchema extends Schema.Schema.AnyNoContext | undefined = undefined,
  TErrorSchema extends Schema.Schema.AnyNoContext | undefined = undefined,
> {
  readonly definition: {
    readonly inputSchema: TInputSchema
    readonly outputSchema: TOutputSchema
  }
  readonly emissionSchema?: TEmissionSchema
  readonly errorSchema?: TErrorSchema
}

type EncodedOrUnknown<TSchema extends Schema.Schema.AnyNoContext | undefined> =
  TSchema extends Schema.Schema.AnyNoContext ? Schema.Schema.Encoded<TSchema> : unknown

type EncodedOrToolError<TSchema extends Schema.Schema.AnyNoContext | undefined> =
  TSchema extends Schema.Schema.AnyNoContext ? Schema.Schema.Encoded<TSchema> & ToolError : ToolError

// ── defineStateModel (curried) ───────────────────────────────────────

/**
 * Curried state model definition.
 *
 * First call binds the tool (for type inference of input/output/emission/error).
 * Second call provides the state schema and config.
 *
 * ```ts
 * const shellState = defineStateModel(shellTool)({
 *   state: ShellStateSchema,
 *   initial: { lastExitCode: Option.none() },
 *   reduce: (state, event) => { ... }
 * })
 * ```
 */
export function defineStateModel<
  TInput,
  TOutput,
  TEmission,
  TError extends ToolError = never,
>(
  _tool: ToolTypeCarrier<TInput, TOutput, TEmission, TError>,
): <TStateSchema extends Schema.Schema.AnyNoContext>(config: {
  readonly state: TStateSchema
  readonly initial: Omit<Schema.Schema.Type<TStateSchema>, 'phase' | 'errorMessage'>
  readonly reduce: (
    state: Schema.Schema.Type<TStateSchema>,
    event: ToolLifecycleEvent<TInput, TOutput, TEmission, TError>,
  ) => Schema.Schema.Type<TStateSchema>
}) => StateModel<TStateSchema, TInput, TOutput, TEmission, TError>

export function defineStateModel<
  TInputSchema extends Schema.Schema.AnyNoContext,
  TOutputSchema extends Schema.Schema.AnyNoContext,
  TEmissionSchema extends Schema.Schema.AnyNoContext | undefined = undefined,
  TErrorSchema extends Schema.Schema.AnyNoContext | undefined = undefined,
>(
  _tool: ToolSchemaCarrier<TInputSchema, TOutputSchema, TEmissionSchema, TErrorSchema>,
): <TStateSchema extends Schema.Schema.AnyNoContext>(config: {
  readonly state: TStateSchema
  readonly initial: Omit<Schema.Schema.Type<TStateSchema>, 'phase' | 'errorMessage'>
  readonly reduce: (
    state: Schema.Schema.Type<TStateSchema>,
    event: ToolLifecycleEvent<
      Schema.Schema.Encoded<TInputSchema>,
      Schema.Schema.Encoded<TOutputSchema>,
      EncodedOrUnknown<TEmissionSchema>,
      EncodedOrToolError<TErrorSchema>
    >,
  ) => Schema.Schema.Type<TStateSchema>
}) => StateModel<
  TStateSchema,
  Schema.Schema.Encoded<TInputSchema>,
  Schema.Schema.Encoded<TOutputSchema>,
  EncodedOrUnknown<TEmissionSchema>,
  EncodedOrToolError<TErrorSchema>
>

export function defineStateModel<
  TInput,
  TOutput,
  TEmission,
  TError extends ToolError = never,
>(
  _tool: ToolTypeCarrier<TInput, TOutput, TEmission, TError> | ToolSchemaCarrier<any, any, any, any>,
): <TStateSchema extends Schema.Schema.AnyNoContext>(config: {
  readonly state: TStateSchema
  readonly initial: Omit<Schema.Schema.Type<TStateSchema>, 'phase' | 'errorMessage'>
  readonly reduce: (
    state: Schema.Schema.Type<TStateSchema>,
    event: ToolLifecycleEvent<TInput, TOutput, TEmission, TError>,
  ) => Schema.Schema.Type<TStateSchema>
}) => StateModel<TStateSchema, TInput, TOutput, TEmission, TError> {
  return <TStateSchema extends Schema.Schema.AnyNoContext>(config: {
    readonly state: TStateSchema
    readonly initial: Omit<Schema.Schema.Type<TStateSchema>, 'phase' | 'errorMessage'>
    readonly reduce: (
      state: Schema.Schema.Type<TStateSchema>,
      event: ToolLifecycleEvent<TInput, TOutput, TEmission, TError>,
    ) => Schema.Schema.Type<TStateSchema>
  }): StateModel<TStateSchema, TInput, TOutput, TEmission, TError> => {
    const initial = Object.freeze({
      phase: "streaming" as const,
      errorMessage: Option.none<string>(),
      ...config.initial,
    }) as Schema.Schema.Type<TStateSchema>

    return { stateSchema: config.state, initial, reduce: config.reduce }
  }
}
