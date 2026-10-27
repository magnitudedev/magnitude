import { Schema } from "effect"

/** A JSON column's encoded fields. */
export const Fields = Schema.Record({ key: Schema.String, value: Schema.Unknown })

/** Fields holding none of `owned`: the row's own columns hold those values, so the JSON may not repeat them. */
export const FieldsWithout = (...owned: ReadonlyArray<string>) => Fields.pipe(Schema.filter((fields) => {
  const repeated = owned.filter((key) => Object.hasOwn(fields, key))
  return repeated.length === 0 || `The row's columns own ${repeated.join(", ")}`
}))
