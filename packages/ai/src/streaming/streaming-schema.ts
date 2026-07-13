import { ParseResult, Schema, SchemaAST as AST } from "effect"
import type { Effect } from "effect"
import {
  inspectSchemaShape,
  type SchemaShape,
  type SchemaShapeIndex,
  type SchemaShapeRef,
} from "@magnitudedev/utils/schema"
import type { ParsedValue } from "./types"
import { parsedValueToJson } from "./values"

export type StreamingSchemaResult<A> =
  | { readonly _tag: "Incomplete" }
  | { readonly _tag: "Complete"; readonly value: A }

export type StreamingSchema<A> = Schema.Schema<StreamingSchemaResult<A>, ParsedValue, never>

type CompleteSchema<A> = Schema.Schema<A, unknown, never>

interface DerivedStreamingSchema<A = unknown> {
  readonly completeSchema: CompleteSchema<A> | null
  readonly streamingSchema: StreamingSchema<A>
  childForObjectKey?(key: string): DerivedStreamingSchema | null
  childForArrayIndex?(index: number): DerivedStreamingSchema | null
}

type StreamingDecode<A> = (
  node: ParsedValue,
  options: AST.ParseOptions,
  ast: AST.Declaration,
) => Effect.Effect<StreamingSchemaResult<A>, ParseResult.ParseIssue, never>

function incomplete<A>(): StreamingSchemaResult<A> {
  return { _tag: "Incomplete" }
}

function complete<A>(value: A): StreamingSchemaResult<A> {
  return { _tag: "Complete", value }
}

function schemaFromAST(ast: AST.AST): CompleteSchema<unknown> {
  return Schema.make(ast) as CompleteSchema<unknown>
}

function isComplete(node: ParsedValue): boolean {
  return node.state === "complete"
}

function isObjectRecord(input: unknown): input is Record<string, unknown> {
  return typeof input === "object" && input !== null
}

function isCompletionState(input: unknown): input is "complete" | "incomplete" {
  return input === "complete" || input === "incomplete"
}

function isParsedValue(input: unknown): input is ParsedValue {
  if (!isObjectRecord(input) || typeof input._tag !== "string") return false

  switch (input._tag) {
    case "string":
    case "number":
      return typeof input.value === "string" && isCompletionState(input.state)
    case "boolean":
      return typeof input.value === "boolean" && input.state === "complete"
    case "null":
      return input.state === "complete"
    case "object":
      return isCompletionState(input.state)
        && Array.isArray(input.entries)
        && input.entries.every((entry) =>
          Array.isArray(entry)
          && entry.length === 2
          && typeof entry[0] === "string"
          && isParsedValue(entry[1])
        )
    case "array":
      return isCompletionState(input.state)
        && Array.isArray(input.items)
        && input.items.every(isParsedValue)
    default:
      return false
  }
}

function declareStreamingSchema<A>(decode: StreamingDecode<A>): StreamingSchema<A> {
  return Schema.declare<StreamingSchemaResult<A>, ParsedValue, readonly []>([], {
    decode: () => (input, options, ast) => {
      if (!isParsedValue(input)) {
        return ParseResult.fail(new ParseResult.Type(ast, input, "Expected ParsedValue"))
      }
      return decode(input, options, ast)
    },
    encode: () => (input, _options, ast) =>
      ParseResult.fail(new ParseResult.Forbidden(ast, input, "Streaming schemas are decode-only")),
  })
}

function decodeComplete<A>(
  schema: CompleteSchema<A>,
  node: ParsedValue,
): Effect.Effect<StreamingSchemaResult<A>, ParseResult.ParseIssue, never> {
  const result = ParseResult.decodeUnknownEither(schema)(parsedValueToJson(node))
  if (result._tag === "Left") return ParseResult.fail(result.left)
  return ParseResult.succeed(complete(result.right))
}

function duplicateKeyIssue(
  key: string,
  node: Extract<ParsedValue, { readonly _tag: "object" }>,
): ParseResult.ParseIssue {
  return new ParseResult.Pointer(
    key,
    node,
    new ParseResult.Type(AST.unknownKeyword, node, `Duplicate object key "${key}"`),
  )
}

function findDuplicateKeyIssue(
  node: Extract<ParsedValue, { readonly _tag: "object" }>,
): ParseResult.ParseIssue | null {
  const seen = new Set<string>()
  for (const [key, value] of node.entries) {
    if (seen.has(key) && value.state === "complete") {
      return duplicateKeyIssue(key, node)
    }
    seen.add(key)
  }
  return null
}

function decodeChild(
  child: DerivedStreamingSchema,
  value: ParsedValue,
  path: PropertyKey,
  actual: ParsedValue,
): ParseResult.ParseIssue | null {
  const result = ParseResult.decodeUnknownEither(child.streamingSchema)(value)
  if (result._tag === "Right") return null
  return new ParseResult.Pointer(path, actual, result.left)
}

class UnknownStreamingSchema implements DerivedStreamingSchema {
  readonly completeSchema = null
  readonly streamingSchema: StreamingSchema<unknown> = declareStreamingSchema<unknown>(() =>
    ParseResult.succeed(incomplete()),
  )

  childForObjectKey(_key: string): DerivedStreamingSchema {
    return this
  }

  childForArrayIndex(_index: number): DerivedStreamingSchema {
    return this
  }
}

const UNKNOWN_STREAMING_SCHEMA = new UnknownStreamingSchema()

class ScalarStreamingSchema<A> implements DerivedStreamingSchema<A> {
  readonly streamingSchema: StreamingSchema<A>

  constructor(readonly completeSchema: CompleteSchema<A>) {
    this.streamingSchema = declareStreamingSchema((node) =>
      isComplete(node) ? decodeComplete(this.completeSchema, node) : ParseResult.succeed(incomplete()),
    )
  }
}

class RefStreamingSchema<A> implements DerivedStreamingSchema<A> {
  readonly streamingSchema: StreamingSchema<A>

  constructor(
    readonly completeSchema: CompleteSchema<A>,
    private readonly target: DerivedStreamingSchema<any>,
  ) {
    this.streamingSchema = declareStreamingSchema((node) => {
      const result = ParseResult.decodeUnknownEither(this.target.streamingSchema)(node)
      if (result._tag === "Left") return ParseResult.fail(result.left)
      if (result.right._tag === "Incomplete") return ParseResult.succeed(incomplete())
      return decodeComplete(this.completeSchema, node)
    })
  }

  childForObjectKey(key: string): DerivedStreamingSchema | null {
    return this.target.childForObjectKey?.(key) ?? null
  }

  childForArrayIndex(index: number): DerivedStreamingSchema | null {
    return this.target.childForArrayIndex?.(index) ?? null
  }
}

class ObjectStreamingSchema<A> implements DerivedStreamingSchema<A> {
  readonly streamingSchema: StreamingSchema<A>

  constructor(
    readonly completeSchema: CompleteSchema<A>,
    private readonly properties: ReadonlyMap<PropertyKey, DerivedStreamingSchema<any>>,
    private readonly indexValue: DerivedStreamingSchema<any> | null,
  ) {
    this.streamingSchema = declareStreamingSchema((node) => {
      if (node._tag !== "object") {
        return isComplete(node) ? decodeComplete(this.completeSchema, node) : ParseResult.succeed(incomplete())
      }

      const duplicateIssue = findDuplicateKeyIssue(node)
      if (duplicateIssue) return ParseResult.fail(duplicateIssue)

      for (const [key, value] of node.entries) {
        const issue = decodeChild(this.childForObjectKey(key), value, key, node)
        if (issue) return ParseResult.fail(issue)
      }

      return node.state === "complete"
        ? decodeComplete(this.completeSchema, node)
        : ParseResult.succeed(incomplete())
    })
  }

  childForObjectKey(key: string): DerivedStreamingSchema {
    return this.properties.get(key) ?? this.indexValue ?? UNKNOWN_STREAMING_SCHEMA
  }
}

class TupleStreamingSchema<A> implements DerivedStreamingSchema<A> {
  readonly streamingSchema: StreamingSchema<A>

  constructor(
    readonly completeSchema: CompleteSchema<A>,
    private readonly elements: readonly DerivedStreamingSchema<any>[],
    private readonly rest: DerivedStreamingSchema<any> | null,
  ) {
    this.streamingSchema = declareStreamingSchema((node) => {
      if (node._tag !== "array") {
        return isComplete(node) ? decodeComplete(this.completeSchema, node) : ParseResult.succeed(incomplete())
      }

      for (let index = 0; index < node.items.length; index += 1) {
        const child = this.childForArrayIndex(index)
        if (!child) continue

        const issue = decodeChild(child, node.items[index], index, node)
        if (issue) return ParseResult.fail(issue)
      }

      return node.state === "complete"
        ? decodeComplete(this.completeSchema, node)
        : ParseResult.succeed(incomplete())
    })
  }

  childForArrayIndex(index: number): DerivedStreamingSchema | null {
    return this.elements[index] ?? this.rest
  }
}

class UnionStreamingSchema<A> implements DerivedStreamingSchema<A> {
  readonly streamingSchema: StreamingSchema<A>

  constructor(
    readonly completeSchema: CompleteSchema<A>,
    private readonly branches: readonly DerivedStreamingSchema<any>[],
  ) {
    this.streamingSchema = declareStreamingSchema((node) => {
      switch (node._tag) {
        case "object": {
          const duplicateIssue = findDuplicateKeyIssue(node)
          if (duplicateIssue) return ParseResult.fail(duplicateIssue)

          for (const [key, value] of node.entries) {
            const child = this.childForObjectKey(key)
            if (!child) continue

            const issue = decodeChild(child, value, key, node)
            if (issue) return ParseResult.fail(issue)
          }
          break
        }
        case "array":
          for (let index = 0; index < node.items.length; index += 1) {
            const child = this.childForArrayIndex(index)
            if (!child) continue

            const issue = decodeChild(child, node.items[index], index, node)
            if (issue) return ParseResult.fail(issue)
          }
          break
        case "string":
        case "number":
          if (node.state !== "complete") return ParseResult.succeed(incomplete())
          break
        case "boolean":
        case "null":
          break
      }

      return node.state === "complete"
        ? decodeComplete(this.completeSchema, node)
        : ParseResult.succeed(incomplete())
    })
  }

  childForObjectKey(key: string): DerivedStreamingSchema | null {
    return combineUnionChildren(
      this.branches.flatMap((branch) => {
        const child = branch.childForObjectKey?.(key) ?? null
        return child === null ? [] : [child]
      }),
    )
  }

  childForArrayIndex(index: number): DerivedStreamingSchema | null {
    return combineUnionChildren(
      this.branches.flatMap((branch) => {
        const child = branch.childForArrayIndex?.(index) ?? null
        return child === null ? [] : [child]
      }),
    )
  }
}

function combineUnionChildren(children: readonly DerivedStreamingSchema<any>[]): DerivedStreamingSchema<any> | null {
  if (children.length === 0) return null
  if (children.some((child) => child.completeSchema === null)) return UNKNOWN_STREAMING_SCHEMA
  if (children.length === 1) return children[0]

  return new UnionStreamingSchema(
    schemaFromAST(AST.Union.make(children.map((child) => child.completeSchema!.ast))),
    children,
  )
}

function completeSchemaFromNode<A>(node: SchemaShape): CompleteSchema<A> {
  return node.schema as CompleteSchema<A>
}

function deriveShapeRef<A>(
  shape: SchemaShapeIndex,
  ref: SchemaShapeRef,
  memo: Map<string, DerivedStreamingSchema<any>>,
  inProgress: Set<string>,
): DerivedStreamingSchema<A> {
  const existing = memo.get(ref.id)
  if (existing) return existing as DerivedStreamingSchema<A>

  if (inProgress.has(ref.id)) {
    return UNKNOWN_STREAMING_SCHEMA as DerivedStreamingSchema<A>
  }

  const node = shape.get(ref)
  const completeSchema = completeSchemaFromNode<A>(node)

  inProgress.add(ref.id)
  try {
    let derived: DerivedStreamingSchema<A>
    switch (node.kind) {
      case "object": {
        const properties = new Map<PropertyKey, DerivedStreamingSchema<any>>()
        for (const field of node.fields) {
          properties.set(field.name, deriveShapeRef(shape, field.value, memo, inProgress))
        }

        const indexValue = node.indexes[0]
          ? deriveShapeRef(shape, node.indexes[0].value, memo, inProgress)
          : null

        derived = new ObjectStreamingSchema(completeSchema, properties, indexValue)
        break
      }
      case "array": {
        derived = new TupleStreamingSchema(
          completeSchema,
          [],
          deriveShapeRef(shape, node.element, memo, inProgress),
        )
        break
      }
      case "tuple": {
        const elements = node.elements.map((element) =>
          deriveShapeRef(shape, element.value, memo, inProgress)
        )
        const rest = node.rest
          ? deriveShapeRef(shape, node.rest, memo, inProgress)
          : null

        derived = new TupleStreamingSchema(completeSchema, elements, rest)
        break
      }
      case "union":
        derived = new UnionStreamingSchema(
          completeSchema,
          node.members.map((member) => deriveShapeRef(shape, member, memo, inProgress)),
        )
        break
      case "alias":
        derived = new RefStreamingSchema(
          completeSchema,
          deriveShapeRef(shape, node.target, memo, inProgress),
        )
        break
      case "scalar":
      case "literal":
      case "enum":
      case "opaque":
        derived = new ScalarStreamingSchema(completeSchema)
        break
    }

    memo.set(ref.id, derived)
    return derived
  } finally {
    inProgress.delete(ref.id)
  }
}

export function deriveStreamingSchema<A, I>(schema: Schema.Schema<A, I, never>): StreamingSchema<A> {
  const shape = inspectSchemaShape(schema as Schema.Schema.AnyNoContext)
  return deriveShapeRef<A>(
    shape,
    shape.root,
    new Map(),
    new Set(),
  ).streamingSchema
}
