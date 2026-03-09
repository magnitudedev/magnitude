/**
 * @magnitudedev/tools — Tool Contract
 *
 * Pure contract for defining tools. A tool is: name + description + schemas + execute Effect.
 * Knows nothing about sandboxes, groups, agents, or runtimes.
 */

import { Effect } from "effect"
import { Schema } from "@effect/schema"
import type { ToolBindings } from "./bindings"

// =============================================================================
// Tool Interface
// =============================================================================

/**
 * A tool: a named, schema-validated function that returns an Effect.
 *
 * Dependencies are declared via the `R` type parameter using Effect's DI system.
 * Whoever executes the tool provides layers for `R`.
 *
 * @typeParam TName - Literal name of the tool (e.g. 'read', 'shell')
 * @typeParam TInput - Input type (derived from inputSchema)
 * @typeParam TOutput - Output type (derived from outputSchema)
 * @typeParam TError - Error type (derived from errorSchema)
 * @typeParam R - Effect requirements — dependencies the tool needs
 * @typeParam TBindings - Preserved binding literal type (inferred from `as const` bindings)
 */
export interface Tool<
  TName extends string = string,
  TInput = unknown,
  TOutput = unknown,
  TError = never,
  R = never,
  TBindings extends ToolBindings<TInput, TOutput> = ToolBindings<TInput, TOutput>,
> {
  readonly name: TName
  readonly description: string
  readonly inputSchema: Schema.Schema<TInput>
  readonly outputSchema: Schema.Schema<TOutput>
  readonly errorSchema?: Schema.Schema.All
  readonly execute: (input: TInput) => Effect.Effect<TOutput, TError, R>

  /**
   * Maps positional arguments to struct fields.
   * Enables call syntax: tool(arg1, arg2) instead of tool({ field1: arg1, field2: arg2 })
   */
  readonly argMapping?: ReadonlyArray<string>

  /**
   * Tool group for sandbox namespacing.
   * Tools with the same group share a namespace (e.g., group 'fs' → fs.read, fs.write).
   * Group 'default' makes tools callable without prefix.
   * No group → standalone global.
   */
  readonly group?: string

  /** Strategy-specific bindings for how this tool is injected into LLM interactions. */
  readonly bindings: TBindings
}

// =============================================================================
// Tool Namespace (type utilities)
// =============================================================================

export namespace Tool {
  /** Extract the name type from a Tool */
  export type Name<T> = T extends Tool<infer N, infer _I, infer _O, infer _E, infer _R, infer _B> ? N : never

  /** Extract the input type from a Tool */
  export type Input<T> = T extends Tool<infer _N, infer I, infer _O, infer _E, infer _R, infer _B> ? I : never

  /** Extract the output type from a Tool */
  export type Output<T> = T extends Tool<infer _N, infer _I, infer O, infer _E, infer _R, infer _B> ? O : never

  /** Extract the error type from a Tool */
  export type Errors<T> = T extends Tool<infer _N, infer _I, infer _O, infer E, infer _R, infer _B> ? E : never

  /** Extract the requirements type from a Tool */
  export type Requirements<T> = T extends Tool<infer _N, infer _I, infer _O, infer _E, infer R, infer _B> ? R : never

  /** Extract the bindings type from a Tool */
  export type Bindings<T> = T extends Tool<infer _N, infer _I, infer _O, infer _E, infer _R, infer B> ? B : never

  /** Extract the xmlInput binding type from a Tool */
  export type XmlInputBinding<T> = NonNullable<Bindings<T>['xmlInput']>

  /** Extract combined requirements from an array of tools */
  export type CombinedRequirements<T extends ReadonlyArray<Any>> =
    T extends readonly [] ? never :
    T[number] extends Tool<infer _N, infer _I, infer _O, infer _E, infer R, infer _B> ? R : never

  /**
   * Any tool regardless of type parameters.
   * Uses `any` to bypass variance issues with function parameters and Schema types.
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  export type Any = Tool<any, any, any, any, any, any>

  /** @internal Runtime marker to make namespace a value (required for isolatedModules) */
  export const _tag = "Tool" as const
}

// =============================================================================
// Tool Factory
// =============================================================================

/**
 * Configuration for creating a tool.
 */
export interface ToolConfig<
  TName extends string,
  InputSchema extends Schema.Schema.AnyNoContext,
  OutputSchema extends Schema.Schema.AnyNoContext,
  ErrorSchema extends Schema.Schema.AnyNoContext | undefined = undefined,
  R = never,
  TBindings extends ToolBindings<Schema.Schema.Type<InputSchema>, Schema.Schema.Type<OutputSchema>> = ToolBindings<Schema.Schema.Type<InputSchema>, Schema.Schema.Type<OutputSchema>>,
> {
  readonly name: TName
  readonly description: string
  readonly inputSchema: InputSchema
  readonly outputSchema: OutputSchema
  readonly errorSchema?: ErrorSchema
  readonly argMapping?: ReadonlyArray<string>
  readonly group?: string
  readonly execute: (
    input: Schema.Schema.Type<InputSchema>
  ) => Effect.Effect<
    Schema.Schema.Type<OutputSchema>,
    ErrorSchema extends Schema.Schema.AnyNoContext ? Schema.Schema.Type<ErrorSchema> : never,
    R
  >

  /** Strategy-specific bindings. Use `as const` to preserve literal types for typed events. */
  readonly bindings: TBindings
}

/**
 * Create a tool from a configuration object.
 *
 * @example
 * ```ts
 * const addTool = createTool({
 *   name: 'add',
 *   description: 'Add two numbers',
 *   inputSchema: Schema.Struct({ a: Schema.Number, b: Schema.Number }),
 *   outputSchema: Schema.Number,
 *   execute: ({ a, b }) => Effect.succeed(a + b)
 * })
 * ```
 */
export function createTool<
  TName extends string,
  InputSchema extends Schema.Schema.AnyNoContext,
  OutputSchema extends Schema.Schema.AnyNoContext,
  ErrorSchema extends Schema.Schema.AnyNoContext | undefined = undefined,
  R = never,
  const TBindings extends ToolBindings<Schema.Schema.Type<InputSchema>, Schema.Schema.Type<OutputSchema>> = ToolBindings<Schema.Schema.Type<InputSchema>, Schema.Schema.Type<OutputSchema>>,
>(
  config: ToolConfig<TName, InputSchema, OutputSchema, ErrorSchema, R, TBindings>
): Tool<
  TName,
  Schema.Schema.Type<InputSchema>,
  Schema.Schema.Type<OutputSchema>,
  ErrorSchema extends Schema.Schema.AnyNoContext ? Schema.Schema.Type<ErrorSchema> : never,
  R,
  TBindings
> {
  return {
    name: config.name,
    description: config.description,
    inputSchema: config.inputSchema,
    outputSchema: config.outputSchema,
    errorSchema: config.errorSchema,
    argMapping: config.argMapping,
    group: config.group,
    execute: config.execute,
    bindings: config.bindings,
  } as Tool<
    TName,
    Schema.Schema.Type<InputSchema>,
    Schema.Schema.Type<OutputSchema>,
    ErrorSchema extends Schema.Schema.AnyNoContext ? Schema.Schema.Type<ErrorSchema> : never,
    R,
    TBindings
  >
}
