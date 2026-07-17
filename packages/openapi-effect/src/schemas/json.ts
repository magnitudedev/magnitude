import { Schema } from "effect";

export type JsonPrimitive = string | number | boolean | null;

export type JsonValue =
  | JsonPrimitive
  | readonly JsonValue[]
  | { readonly [key: string]: JsonValue };

export const JsonPrimitive = Schema.Union(
  Schema.String,
  Schema.Number,
  Schema.Boolean,
  Schema.Null
);

export const JsonValue: Schema.Schema<JsonValue, JsonValue> = Schema.suspend(
  () =>
    Schema.Union(
      JsonPrimitive,
      Schema.Array(JsonValue),
      Schema.Record({ key: Schema.String, value: JsonValue })
    )
);

export const JsonObject = Schema.Record({
  key: Schema.String,
  value: JsonValue,
});
export type JsonObject = typeof JsonObject.Type;
