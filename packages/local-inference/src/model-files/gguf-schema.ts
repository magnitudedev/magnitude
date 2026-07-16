import { Schema } from "effect"

export const GgufTypedMetadataEntry = Schema.Struct({ value: Schema.Unknown })
export const GgufTypedMetadata = Schema.Record({ key: Schema.String, value: GgufTypedMetadataEntry })
export type GgufTypedMetadata = Schema.Schema.Type<typeof GgufTypedMetadata>

export const GgufReaderDocument = Schema.Struct({
  typedMetadata: GgufTypedMetadata,
  parameterCount: Schema.optional(Schema.Union(Schema.Number, Schema.BigIntFromSelf)),
})
export type GgufReaderDocument = Schema.Schema.Type<typeof GgufReaderDocument>
