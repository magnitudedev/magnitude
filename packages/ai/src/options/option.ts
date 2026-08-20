import { Array as EffectArray, Data, Effect, Option as EffectOption, Schema } from "effect"
import type * as ParseResult from "effect/ParseResult"
import type { JsonRecord, JsonValue } from "@magnitudedev/utils/schema"
import { toCauseInfo, type CauseInfo } from "../errors/failure"

export class OptionContributionError extends Data.TaggedError("OptionContributionError")<{
  readonly option: string
  readonly failure:
    | { readonly _tag: "OptionMappingFailed"; readonly cause: CauseInfo }
    | { readonly _tag: "OptionEncodingFailed"; readonly error: ParseResult.ParseError }
}> {}

class OptionMapperError extends Data.TaggedError("OptionMapperError")<{
  readonly cause: CauseInfo
}> {}

export interface OptionDefErased {
  readonly _tag: "OptionDef"
  readonly required: boolean
  readonly default?: unknown
  readonly contributionSchema: Schema.Schema.Any
  // Dynamic application erases the caller value type; concrete definitions below remain exact.
  readonly encodeContribution: (
    value: any,
  ) => Effect.Effect<JsonRecord, OptionMapperError | ParseResult.ParseError>
}

export interface OptionDefConcrete<
  TValue,
  TContributionValue extends object,
  TContribution extends JsonRecord,
  TRequired extends boolean,
> extends OptionDefErased {
  readonly required: TRequired
  readonly default?: TValue
  readonly contributionSchema: Schema.Schema<TContributionValue, TContribution, never>
  readonly encodeContribution: (
    value: TValue,
  ) => Effect.Effect<TContribution, OptionMapperError | ParseResult.ParseError>
}

export type OptionDef<
  TValue = never,
  TContributionValue extends object = never,
  TContribution extends JsonRecord = never,
  TRequired extends boolean = false,
> = [TValue] extends [never]
  ? OptionDefErased
  : OptionDefConcrete<TValue, TContributionValue, TContribution, TRequired>

export type ExtractValue<T> = T extends OptionDefConcrete<infer V, any, any, any> ? V : never

export type ExtractRequired<T> = T extends OptionDefConcrete<any, any, any, infer R> ? R : false

export type RequiredKeys<T extends Record<string, OptionDef>> = {
  [K in keyof T]: ExtractRequired<T[K]> extends true ? K : never
}[keyof T]

export type OptionalKeys<T extends Record<string, OptionDef>> = {
  [K in keyof T]: ExtractRequired<T[K]> extends true ? never : K
}[keyof T]

export type InferCallOptions<T extends Record<string, OptionDef>> =
  & { readonly [K in RequiredKeys<T>]: ExtractValue<T[K]> }
  & { readonly [K in OptionalKeys<T>]?: ExtractValue<T[K]> }

const define = <TValue, A extends object, I extends JsonRecord>(
  contributionSchema: Schema.Schema<A, I, never>,
  map: (value: TValue) => A,
  defaultValue?: TValue,
): OptionDefConcrete<TValue, A, I, false> => ({
  _tag: "OptionDef",
  required: false,
  default: defaultValue,
  contributionSchema,
  encodeContribution: (value) => Effect.try({
    try: () => map(value),
    catch: (cause) => new OptionMapperError({ cause: toCauseInfo(cause) }),
  }).pipe(Effect.flatMap(Schema.encode(contributionSchema))),
})

const required = <TValue, A extends object, I extends JsonRecord>(
  contributionSchema: Schema.Schema<A, I, never>,
  map: (value: TValue) => A,
): OptionDefConcrete<TValue, A, I, true> => ({
  _tag: "OptionDef",
  required: true,
  contributionSchema,
  encodeContribution: (value) => Effect.try({
    try: () => map(value),
    catch: (cause) => new OptionMapperError({ cause: toCauseInfo(cause) }),
  }).pipe(Effect.flatMap(Schema.encode(contributionSchema))),
})

const field = <K extends string, A, I extends JsonValue>(
  key: K,
  valueSchema: Schema.Schema<A, I, never>,
  defaultValue?: A,
): OptionDefConcrete<
  A,
  { readonly [P in K]: A },
  { readonly [P in K]: I },
  false
> => {
  const contributionSchema = Schema.Record({
    key: Schema.Literal(key),
    value: valueSchema,
  })
  return define(contributionSchema, (value: A) => ({ [key]: value } as {
    readonly [P in K]: A
  }), defaultValue)
}

const requiredField = <K extends string, A, I extends JsonValue>(
  key: K,
  valueSchema: Schema.Schema<A, I, never>,
): OptionDefConcrete<
  A,
  { readonly [P in K]: A },
  { readonly [P in K]: I },
  true
> => {
  const contributionSchema = Schema.Record({
    key: Schema.Literal(key),
    value: valueSchema,
  })
  return required(contributionSchema, (value: A) => ({ [key]: value } as {
    readonly [P in K]: A
  }))
}

const ignore = <TValue>(): OptionDefConcrete<TValue, {}, {}, false> =>
  define(Schema.Struct({}), (_value: TValue) => ({}))

export const Option = {
  define,
  required,
  field,
  requiredField,
  ignore,
} as const

export function applyOptionDefs<T extends object>(
  defs: Record<string, OptionDefErased>,
  options: T,
): Effect.Effect<readonly JsonRecord[], OptionContributionError> {
  const values = options as Record<string, unknown>
  return Effect.forEach(Object.entries(defs), ([key, def]) => {
    const supplied = values[key]
    const value = supplied === undefined ? def.default : supplied
    if (value === undefined) return Effect.succeed(EffectOption.none<JsonRecord>())
    return def.encodeContribution(value).pipe(
      Effect.map(EffectOption.some),
      Effect.mapError((error) => new OptionContributionError({
        option: key,
        failure: error instanceof OptionMapperError
          ? { _tag: "OptionMappingFailed", cause: error.cause }
          : { _tag: "OptionEncodingFailed", error },
      })),
    )
  }).pipe(Effect.map((contributions) => EffectArray.getSomes(contributions)))
}
