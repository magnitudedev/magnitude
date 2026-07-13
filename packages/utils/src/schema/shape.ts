import { Option, Schema, SchemaAST as AST } from 'effect'
import { isJsonValue, type SchemaShapeJsonValue } from './json-value'

export type { SchemaShapeJsonValue } from './json-value'

export type SchemaShapeChannel = 'encoded' | 'type'

export interface InspectSchemaShapeOptions {
  readonly channel?: SchemaShapeChannel
  readonly maxNodes?: number
  readonly onOpaque?: 'keep' | 'throw'
}

export type SchemaShapeId = string

export interface SchemaShapeIndex {
  readonly root: SchemaShapeRef
  readonly get: (ref: SchemaShapeRef) => SchemaShape
}

export interface SchemaShapeRef {
  readonly id: SchemaShapeId
}

export interface SchemaShapeMeta {
  readonly identifier?: string
  readonly title?: string
  readonly description?: string
  readonly defaultValue?: SchemaShapeJsonValue
}

export type ScalarKind =
  | 'string'
  | 'number'
  | 'boolean'
  | 'null'
  | 'unknown'
  | 'any'
  | 'never'
  | 'void'
  | 'undefined'
  | 'object'
  | 'symbol'
  | 'bigint'

export type SchemaShape =
  | {
    readonly kind: 'scalar'
    readonly scalar: ScalarKind
    readonly schema: Schema.Schema.AnyNoContext
    readonly meta: SchemaShapeMeta
  }
  | {
    readonly kind: 'literal'
    readonly value: string | number | boolean | null
    readonly schema: Schema.Schema.AnyNoContext
    readonly meta: SchemaShapeMeta
  }
  | {
    readonly kind: 'enum'
    readonly cases: readonly {
      readonly name: string
      readonly value: string | number
    }[]
    readonly schema: Schema.Schema.AnyNoContext
    readonly meta: SchemaShapeMeta
  }
  | {
    readonly kind: 'object'
    readonly fields: readonly SchemaShapeField[]
    readonly indexes: readonly SchemaShapeIndexSignature[]
    readonly schema: Schema.Schema.AnyNoContext
    readonly meta: SchemaShapeMeta
  }
  | {
    readonly kind: 'array'
    readonly element: SchemaShapeRef
    readonly schema: Schema.Schema.AnyNoContext
    readonly meta: SchemaShapeMeta
  }
  | {
    readonly kind: 'tuple'
    readonly elements: readonly SchemaShapeTupleElement[]
    readonly rest?: SchemaShapeRef
    readonly schema: Schema.Schema.AnyNoContext
    readonly meta: SchemaShapeMeta
  }
  | {
    readonly kind: 'union'
    readonly members: readonly SchemaShapeRef[]
    readonly schema: Schema.Schema.AnyNoContext
    readonly meta: SchemaShapeMeta
  }
  | {
    readonly kind: 'alias'
    readonly target: SchemaShapeRef
    readonly reason: 'suspend' | 'refinement' | 'transformation' | 'surrogate'
    readonly schema: Schema.Schema.AnyNoContext
    readonly meta: SchemaShapeMeta
  }
  | {
    readonly kind: 'opaque'
    readonly reason: string
    readonly schema: Schema.Schema.AnyNoContext
    readonly meta: SchemaShapeMeta
  }

export interface SchemaShapeField {
  readonly name: string
  readonly optional: boolean
  readonly value: SchemaShapeRef
  readonly meta: SchemaShapeMeta
}

export interface SchemaShapeIndexSignature {
  readonly key: SchemaShapeRef
  readonly value: SchemaShapeRef
}

export interface SchemaShapeTupleElement {
  readonly optional: boolean
  readonly value: SchemaShapeRef
}

export class SchemaShapeInspectionError extends Error {
  constructor(
    message: string,
    readonly schemaIdentifier?: string,
  ) {
    super(schemaIdentifier ? `${message} (${schemaIdentifier})` : message)
    this.name = 'SchemaShapeInspectionError'
  }
}

const DEFAULT_MAX_NODES = 4096

function optionValue<T>(option: Option.Option<T>): T | undefined {
  return Option.isSome(option) ? option.value : undefined
}

function schemaIdentifier(ast: AST.Annotated): string | undefined {
  return optionValue(AST.getIdentifierAnnotation(ast))
    ?? optionValue(AST.getJSONIdentifierAnnotation(ast))
}

function defaultValue(annotated: AST.Annotated): SchemaShapeJsonValue | undefined {
  const annotation = AST.getDefaultAnnotation(annotated)
  if (Option.isNone(annotation)) return undefined

  const value = typeof annotation.value === 'function'
    ? (annotation.value as () => unknown)()
    : annotation.value

  return isJsonValue(value) ? value : undefined
}

function meta(annotated: AST.Annotated): SchemaShapeMeta {
  return {
    identifier: schemaIdentifier(annotated),
    title: optionValue(AST.getTitleAnnotation(annotated)),
    description: optionValue(AST.getDescriptionAnnotation(annotated)),
    defaultValue: defaultValue(annotated),
  }
}

function propertyName(name: PropertyKey): string {
  return typeof name === 'symbol' ? name.toString() : String(name)
}

function schemaFromAst(ast: AST.AST): Schema.Schema.AnyNoContext {
  return Schema.make(ast) as Schema.Schema.AnyNoContext
}

export function inspectSchemaShape(
  schema: Schema.Schema.AnyNoContext,
  options?: InspectSchemaShapeOptions,
): SchemaShapeIndex {
  const channel = options?.channel ?? 'encoded'
  const maxNodes = options?.maxNodes ?? DEFAULT_MAX_NODES
  const onOpaque = options?.onOpaque ?? 'keep'
  const refs = new WeakMap<AST.AST, SchemaShapeRef>()
  const shapes = new Map<SchemaShapeId, SchemaShape>()
  let nextId = 0

  function nextRef(ast: AST.AST): SchemaShapeRef {
    const existing = refs.get(ast)
    if (existing) return existing

    if (nextId >= maxNodes) {
      throw new SchemaShapeInspectionError('Schema shape node limit exceeded', schemaIdentifier(schema.ast))
    }

    const ref = { id: `shape:${nextId}` }
    nextId += 1
    refs.set(ast, ref)
    shapes.set(ref.id, buildShape(ast))
    return ref
  }

  function opaque(ast: AST.AST, reason: string): SchemaShape {
    if (onOpaque === 'throw') {
      throw new SchemaShapeInspectionError(reason, schemaIdentifier(ast))
    }
    return {
      kind: 'opaque',
      reason,
      schema: schemaFromAst(ast),
      meta: meta(ast),
    }
  }

  function alias(
    ast: AST.AST,
    reason: 'suspend' | 'refinement' | 'transformation' | 'surrogate',
    target: AST.AST,
  ): SchemaShape {
    return {
      kind: 'alias',
      reason,
      target: nextRef(target),
      schema: schemaFromAst(ast),
      meta: meta(ast),
    }
  }

  function buildShape(ast: AST.AST): SchemaShape {
    const surrogate = AST.getSurrogateAnnotation(ast)
    if (Option.isSome(surrogate)) return alias(ast, 'surrogate', surrogate.value)

    switch (ast._tag) {
      case 'StringKeyword':
        return { kind: 'scalar', scalar: 'string', schema: schemaFromAst(ast), meta: meta(ast) }
      case 'NumberKeyword':
        return { kind: 'scalar', scalar: 'number', schema: schemaFromAst(ast), meta: meta(ast) }
      case 'BooleanKeyword':
        return { kind: 'scalar', scalar: 'boolean', schema: schemaFromAst(ast), meta: meta(ast) }
      case 'VoidKeyword':
        return { kind: 'scalar', scalar: 'void', schema: schemaFromAst(ast), meta: meta(ast) }
      case 'NeverKeyword':
        return { kind: 'scalar', scalar: 'never', schema: schemaFromAst(ast), meta: meta(ast) }
      case 'UnknownKeyword':
        return { kind: 'scalar', scalar: 'unknown', schema: schemaFromAst(ast), meta: meta(ast) }
      case 'AnyKeyword':
        return { kind: 'scalar', scalar: 'any', schema: schemaFromAst(ast), meta: meta(ast) }
      case 'UndefinedKeyword':
        return { kind: 'scalar', scalar: 'undefined', schema: schemaFromAst(ast), meta: meta(ast) }
      case 'ObjectKeyword':
        return { kind: 'scalar', scalar: 'object', schema: schemaFromAst(ast), meta: meta(ast) }
      case 'SymbolKeyword':
      case 'UniqueSymbol':
        return { kind: 'scalar', scalar: 'symbol', schema: schemaFromAst(ast), meta: meta(ast) }
      case 'BigIntKeyword':
        return { kind: 'scalar', scalar: 'bigint', schema: schemaFromAst(ast), meta: meta(ast) }
      case 'Literal':
        if (ast.literal === null) {
          return { kind: 'scalar', scalar: 'null', schema: schemaFromAst(ast), meta: meta(ast) }
        }
        if (
          typeof ast.literal === 'string'
          || typeof ast.literal === 'number'
          || typeof ast.literal === 'boolean'
        ) {
          return { kind: 'literal', value: ast.literal, schema: schemaFromAst(ast), meta: meta(ast) }
        }
        return opaque(ast, `Unsupported literal ${String(ast.literal)}`)
      case 'Enums':
        return {
          kind: 'enum',
          cases: ast.enums.map(([name, value]) => ({ name, value })),
          schema: schemaFromAst(ast),
          meta: meta(ast),
        }
      case 'TemplateLiteral':
        return { kind: 'scalar', scalar: 'string', schema: schemaFromAst(ast), meta: meta(ast) }
      case 'Refinement':
        return alias(ast, 'refinement', ast.from)
      case 'Transformation':
        return alias(ast, 'transformation', channel === 'encoded' ? ast.from : ast.to)
      case 'Suspend':
        return alias(ast, 'suspend', ast.f())
      case 'TypeLiteral':
        return {
          kind: 'object',
          fields: ast.propertySignatures.map((property) => ({
            name: propertyName(property.name),
            optional: property.isOptional,
            value: nextRef(property.type),
            meta: meta(property),
          })),
          indexes: ast.indexSignatures.map((index) => ({
            key: nextRef(index.parameter),
            value: nextRef(index.type),
          })),
          schema: schemaFromAst(ast),
          meta: meta(ast),
        }
      case 'TupleType': {
        if (ast.elements.length === 0 && ast.rest.length === 1) {
          return {
            kind: 'array',
            element: nextRef(ast.rest[0].type),
            schema: schemaFromAst(ast),
            meta: meta(ast),
          }
        }

        return {
          kind: 'tuple',
          elements: ast.elements.map((element) => ({
            optional: element.isOptional,
            value: nextRef(element.type),
          })),
          rest: ast.rest[0] ? nextRef(ast.rest[0].type) : undefined,
          schema: schemaFromAst(ast),
          meta: meta(ast),
        }
      }
      case 'Union':
        return {
          kind: 'union',
          members: ast.types.map(nextRef),
          schema: schemaFromAst(ast),
          meta: meta(ast),
        }
      case 'Declaration':
        return opaque(ast, 'Effect declaration schema has no generic shape')
    }
  }

  const root = nextRef(schema.ast)
  return {
    root,
    get(ref) {
      const shape = shapes.get(ref.id)
      if (!shape) throw new SchemaShapeInspectionError(`Missing schema shape ${ref.id}`, schemaIdentifier(schema.ast))
      return shape
    },
  }
}
