import { Schema } from 'effect'

export type JsonPrimitive = string | number | boolean | null

export type JsonValue =
  | string
  | number
  | boolean
  | null
  | readonly JsonValue[]
  | { readonly [key: string]: JsonValue }

export type SchemaShapeJsonValue = JsonValue

export const JsonValueSchema: Schema.Schema<JsonValue> = Schema.suspend(() =>
  Schema.Union(
    Schema.String,
    Schema.Number,
    Schema.Boolean,
    Schema.Null,
    Schema.Array(JsonValueSchema),
    Schema.Record({ key: Schema.String, value: JsonValueSchema }),
  ),
).annotations({ identifier: 'JsonValue' })

export type JsonRecord = { readonly [key: string]: JsonValue }
export type JsonObject = JsonRecord

export const JsonRecordSchema: Schema.Schema<JsonRecord> = Schema.Record({
  key: Schema.String,
  value: JsonValueSchema,
})

export type JsonEncoded<S extends Schema.Schema.AnyNoContext> =
  [Schema.Schema.Encoded<S>] extends [JsonValue] ? Schema.Schema.Encoded<S> : never

export type JsonEncodedSchema<S extends Schema.Schema.AnyNoContext> =
  [Schema.Schema.Encoded<S>] extends [JsonValue] ? S : never

export type JsonObjectEncodedSchema<S extends Schema.Schema.AnyNoContext> =
  [Schema.Schema.Encoded<S>] extends [JsonObject] ? S : never

export function defineJsonEncodedSchema<const S extends Schema.Schema.AnyNoContext>(
  schema: JsonEncodedSchema<S>,
): S {
  return schema
}

export function defineJsonObjectEncodedSchema<const S extends Schema.Schema.AnyNoContext>(
  schema: JsonObjectEncodedSchema<S>,
): S {
  return schema
}

export const NoInputSchema = defineJsonObjectEncodedSchema(
  Schema.Record({ key: Schema.String, value: Schema.Never }).annotations({
    identifier: 'NoInput',
    jsonSchema: {
      type: 'object',
      required: [],
      properties: {},
      additionalProperties: false,
    },
  }),
)

export type SchemaEntry = readonly [string, Schema.Schema.AnyNoContext]
export type NonEmptySchemaEntries = readonly [SchemaEntry, ...SchemaEntry[]]
export type SchemaFromEntry<TEntry extends SchemaEntry> =
  TEntry extends readonly [string, infer S extends Schema.Schema.AnyNoContext] ? S : never
export type SchemaFromEntries<TEntries extends NonEmptySchemaEntries> = SchemaFromEntry<TEntries[number]>
export type SchemaMapFromEntries<TEntries extends NonEmptySchemaEntries> = {
  readonly [Entry in TEntries[number] as Entry[0]]: Entry[1]
}
export type JsonEncodedSchemaEntries<TEntries extends NonEmptySchemaEntries> = {
  readonly [Index in keyof TEntries]: TEntries[Index] extends readonly [
    infer Key extends string,
    infer S extends Schema.Schema.AnyNoContext,
  ] ? readonly [Key, JsonEncodedSchema<S>]
    : TEntries[Index]
}

export function defineJsonEncodedSchemaEntries<const TEntries extends NonEmptySchemaEntries>(
  entries: TEntries & JsonEncodedSchemaEntries<TEntries>,
): TEntries {
  return entries
}

export function schemaMapFromEntries<const TEntries extends NonEmptySchemaEntries>(
  entries: TEntries,
): SchemaMapFromEntries<TEntries> {
  return Object.fromEntries(entries) as SchemaMapFromEntries<TEntries>
}

export function makeSchemaUnionFromEntries<const TEntries extends NonEmptySchemaEntries>(
  entries: TEntries,
) {
  const values = entries.map(([, schema]) => schema)
  return Schema.Union(...values as [SchemaFromEntries<TEntries>, ...SchemaFromEntries<TEntries>[]])
}

export function isJsonValue(
  value: unknown,
  stack = new WeakSet<object>(),
): value is SchemaShapeJsonValue {
  if (value === null) return true
  switch (typeof value) {
    case 'string':
    case 'number':
    case 'boolean':
      return true
    case 'object':
      break
    default:
      return false
  }

  if (stack.has(value)) return false
  stack.add(value)
  try {
    if (Array.isArray(value)) return value.every((item) => isJsonValue(item, stack))
    if (Object.getOwnPropertySymbols(value).length > 0) return false
    return Object.values(value).every((item) => isJsonValue(item, stack))
  } finally {
    stack.delete(value)
  }
}
