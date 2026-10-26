import { ParseResult, Schema } from "effect"

// The stored and wire JSON convention: plain JSON with `type` discriminators, absent keys for
// absent values and epoch-millisecond times, decoded to `_tag`, `Option` and `DateTime.Utc`.

/** A tagged struct: `_tag` in the runtime, `type` when encoded. */
export const tagged = <Tag extends string, Fields extends Schema.Struct.Fields>(tag: Tag, fields: Fields) =>
  Schema.Struct({ _tag: Schema.tag(tag).pipe(Schema.fromKey("type")), ...fields })

/** `Option` in the runtime, an absent key when encoded. */
export const optional = <A, I, R>(schema: Schema.Schema<A, I, R>) =>
  Schema.optionalWith(schema, { as: "Option", exact: true })

/** `DateTime.Utc` in the runtime, epoch milliseconds when encoded. */
export const Millis = Schema.DateTimeUtcFromNumber

/**
 * A stored type's versions: encodes at the current version and decodes any version it lists, so a
 * breaking change keeps decoding what was stored before it. The encoded side pairs the version
 * with the value encoded at it.
 */
export const versioned = <A, I>(
  current: { readonly version: number; readonly schema: Schema.Schema<A, I> },
  // Each older version has its own encoded form.
  older: ReadonlyArray<{ readonly version: number; readonly schema: Schema.Schema<A, any> }> = []
): Schema.Schema<A, { readonly version: number; readonly value: unknown }> => {
  const versions = new Map<number, Schema.Schema<A, any>>([
    ...older.map(({ version, schema }) => [version, schema] as const),
    [current.version, current.schema],
  ])
  return Schema.transformOrFail(
    Schema.Struct({ version: Schema.Number, value: Schema.Unknown }),
    Schema.typeSchema(current.schema),
    {
      strict: true,
      // The caller's parse options, such as rejecting excess properties, reach the value.
      decode: ({ version, value }, options, ast) => {
        const schema = versions.get(version)
        return schema === undefined
          ? ParseResult.fail(new ParseResult.Type(ast, version, `Version ${version} is not supported`))
          : ParseResult.decodeUnknown(schema, options)(value)
      },
      encode: (value, options) =>
        ParseResult.map(ParseResult.encode(current.schema, options)(value), (encoded) => ({ version: current.version, value: encoded })),
    }
  )
}
