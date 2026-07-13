import { JSONSchema, Option, Schema, SchemaAST as AST } from 'effect'

export type JsonSchema = JSONSchema.JsonSchema7
export type JsonSchemaValue =
  | string
  | number
  | boolean
  | null
  | readonly JsonSchemaValue[]
  | JsonSchemaJsonObject

export interface JsonSchemaJsonObject {
  readonly [key: string]: JsonSchemaValue
}

export type JsonSchemaObject = JSONSchema.JsonSchema7Root & {
  readonly [key: string]: JsonSchemaValue
}

export type JsonSchemaRoot = JsonSchemaObject

export type JsonSchemaTarget =
  | 'jsonSchema7'
  | 'jsonSchema2019-09'
  | 'jsonSchema2020-12'
  | 'openApi3.1'

export interface MakeJsonSchemaOptions {
  readonly target?: JsonSchemaTarget
}

export class JsonSchemaGenerationError extends Error {
  constructor(
    message: string,
    readonly schemaIdentifier?: string,
    readonly cause?: unknown,
  ) {
    super(schemaIdentifier ? `${message} (${schemaIdentifier})` : message)
    this.name = 'JsonSchemaGenerationError'
  }
}

function schemaIdentifier(schema: Schema.Schema.AnyNoContext): string | undefined {
  const identifier = AST.getIdentifierAnnotation(schema.ast)
  if (Option.isSome(identifier)) return identifier.value

  const jsonIdentifier = AST.getJSONIdentifierAnnotation(schema.ast)
  return Option.isSome(jsonIdentifier) ? jsonIdentifier.value : undefined
}

export function makeJsonSchema(
  schema: Schema.Schema.AnyNoContext,
  options?: MakeJsonSchemaOptions,
): JsonSchemaRoot {
  try {
    return JSONSchema.make(schema, options) as JsonSchemaRoot
  } catch (cause) {
    throw new JsonSchemaGenerationError('Failed to generate JSON Schema', schemaIdentifier(schema), cause)
  }
}
