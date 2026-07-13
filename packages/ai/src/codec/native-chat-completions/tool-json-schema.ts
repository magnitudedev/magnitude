import type { Schema } from "effect"
import {
  makeJsonSchema,
  type JsonSchemaJsonObject,
  type JsonSchemaObject,
  type JsonSchemaValue,
} from "@magnitudedev/utils/schema"

function asObject(value: JsonSchemaValue | undefined): JsonSchemaJsonObject | undefined {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return undefined
  return value as JsonSchemaJsonObject
}

function localPointerSegment(segment: string): string {
  return segment.replace(/~1/g, "/").replace(/~0/g, "~")
}

function resolveLocalRef(root: JsonSchemaJsonObject, ref: string): JsonSchemaJsonObject | undefined {
  if (!ref.startsWith("#/")) return undefined

  let current: JsonSchemaValue | undefined = root
  for (const segment of ref.slice(2).split("/")) {
    const object = asObject(current)
    if (!object) return undefined
    current = object[localPointerSegment(segment)]
  }
  return asObject(current)
}

function isObjectParameterSchema(root: JsonSchemaJsonObject, schema: JsonSchemaJsonObject): boolean {
  if (schema.type === "object") return true

  const ref = typeof schema.$ref === "string" ? schema.$ref : undefined
  if (!ref) return false

  const target = resolveLocalRef(root, ref)
  return target?.type === "object"
}

function omitRootMetaSchema(schema: JsonSchemaObject): JsonSchemaObject {
  const { $schema: _metaSchema, ...rest } = schema
  return rest
}

function prepareNativeToolParameterSchema(schema: JsonSchemaObject): JsonSchemaObject {
  if (!isObjectParameterSchema(schema, schema)) {
    throw new Error("Native tool parameters must be encoded as a JSON object schema")
  }
  return omitRootMetaSchema(schema)
}

export function makeNativeToolParametersJsonSchema(
  schema: Schema.Schema.AnyNoContext,
): JsonSchemaObject {
  return prepareNativeToolParameterSchema(makeJsonSchema(schema, { target: "jsonSchema7" }))
}
