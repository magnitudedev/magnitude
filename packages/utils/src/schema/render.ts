import { Option, Schema, SchemaAST as AST } from 'effect'
import {
  inspectSchemaShape,
  type SchemaShape,
  type SchemaShapeIndex,
  type SchemaShapeRef,
} from './shape'
import { isJsonValue, type SchemaShapeJsonValue } from './json-value'

interface RenderSchemaOptions {
  readonly maxNodes?: number
}

interface RenderContext {
  readonly shape: SchemaShapeIndex
}

interface ParamInfo {
  readonly name: string
  readonly optional: boolean
  readonly type: string
  readonly description: string | undefined
  readonly defaultValue: SchemaShapeJsonValue | undefined
}

export interface RenderableSchemaTool {
  readonly name: string
  readonly description: string
  readonly inputSchema: Schema.Schema.AnyNoContext
  readonly outputSchema: Schema.Schema.Any
}

function isNoiseDescription(desc: string | undefined): boolean {
  if (!desc) return true
  return /^a (string|number|boolean|unknown|void|never|object|array)/.test(desc)
}

function formatDefaultValue(value: SchemaShapeJsonValue): string {
  return JSON.stringify(value)
}

function extractDefaultsFromTransformation(ast: AST.AST): Map<string, SchemaShapeJsonValue> {
  const defaults = new Map<string, SchemaShapeJsonValue>()
  if (ast._tag !== 'Transformation') return defaults
  if (ast.transformation._tag !== 'TypeLiteralTransformation') return defaults

  for (const pst of ast.transformation.propertySignatureTransformations) {
    const propName = String(pst.from)
    try {
      const result = pst.decode(Option.none())
      if (Option.isSome(result) && isJsonValue(result.value)) {
        defaults.set(propName, result.value)
      }
    } catch {
      // Optional default extraction is best-effort only.
    }
  }
  return defaults
}

function nodeFor(ctx: RenderContext, ref: SchemaShapeRef): SchemaShape {
  return ctx.shape.get(ref)
}

function recursiveName(node: SchemaShape): string {
  return node.meta.identifier ?? '<recursive>'
}

function isUndefinedRef(ref: SchemaShapeRef, ctx: RenderContext, seen = new Set<string>()): boolean {
  if (seen.has(ref.id)) return false
  seen.add(ref.id)
  const node = nodeFor(ctx, ref)
  if (node.kind === 'scalar') return node.scalar === 'undefined'
  if (node.kind === 'alias') return isUndefinedRef(node.target, ctx, seen)
  return false
}

function renderObject(node: Extract<SchemaShape, { readonly kind: 'object' }>, depth: number, ctx: RenderContext, stack: Set<string>): string {
  const parts: string[] = []

  for (const field of node.fields) {
    const opt = field.optional ? '?' : ''
    parts.push(`${field.name}${opt}: ${typeToString(field.value, field.optional, depth + 1, ctx, stack)}`)
  }

  for (const index of node.indexes) {
    const key = typeToString(index.key, false, depth + 1, ctx, stack)
    const value = typeToString(index.value, false, depth + 1, ctx, stack)
    parts.push(`[key: ${key}]: ${value}`)
  }

  if (depth > 0) return `{ ${parts.join(', ')} }`
  return `{\n${parts.map((part) => `\t${part}`).join(',\n')}\n}`
}

function typeToString(
  ref: SchemaShapeRef,
  isOptional: boolean,
  depth: number,
  ctx: RenderContext,
  stack: Set<string>,
): string {
  const node = nodeFor(ctx, ref)
  if (stack.has(ref.id)) return recursiveName(node)
  if (node.meta.identifier) return node.meta.identifier

  stack.add(ref.id)
  try {
    switch (node.kind) {
      case 'scalar':
        return node.scalar
      case 'literal':
        return JSON.stringify(node.value)
      case 'enum':
        return node.cases.map((entry) => JSON.stringify(entry.value)).join(' | ')
      case 'object':
        return renderObject(node, depth, ctx, stack)
      case 'array':
        return `${typeToString(node.element, false, depth, ctx, stack)}[]`
      case 'tuple': {
        const elements = node.elements.map((element) => typeToString(element.value, element.optional, depth, ctx, stack))
        const rest = node.rest
          ? [`...${typeToString(node.rest, false, depth, ctx, stack)}[]`]
          : []
        return `[${[...elements, ...rest].join(', ')}]`
      }
      case 'union': {
        const nonUndefined = node.members.filter((member) => !isUndefinedRef(member, ctx))
        if (nonUndefined.length === 1 && isOptional) {
          return typeToString(nonUndefined[0], false, depth, ctx, stack)
        }

        const rendered = nonUndefined.map((member) => typeToString(member, false, depth, ctx, stack))
        return rendered.join(' | ')
      }
      case 'alias':
        return typeToString(node.target, isOptional, depth, ctx, stack)
      case 'opaque':
        return node.meta.identifier ?? 'unknown'
    }
  } finally {
    stack.delete(ref.id)
  }
}

function buildComment(description: string | undefined, defaultValue: SchemaShapeJsonValue | undefined): string {
  const cleanDesc = isNoiseDescription(description) ? undefined : description
  if (!cleanDesc && defaultValue === undefined) return ''
  const parts: string[] = []
  if (cleanDesc) parts.push(cleanDesc)
  if (defaultValue !== undefined) {
    parts.push(`(default: ${formatDefaultValue(defaultValue)})`)
  }
  return ` // ${parts.join(' ')}`
}

function resolveObject(
  ref: SchemaShapeRef,
  ctx: RenderContext,
  seen = new Set<string>(),
): Extract<SchemaShape, { readonly kind: 'object' }> | null {
  if (seen.has(ref.id)) return null
  seen.add(ref.id)

  const node = nodeFor(ctx, ref)
  if (node.kind === 'object') return node
  if (node.kind === 'alias') return resolveObject(node.target, ctx, seen)
  return null
}

function descriptionFor(ref: SchemaShapeRef, ctx: RenderContext, seen = new Set<string>()): string | undefined {
  if (seen.has(ref.id)) return undefined
  seen.add(ref.id)

  const node = nodeFor(ctx, ref)
  if (node.meta.description) return node.meta.description
  if (node.kind === 'alias') return descriptionFor(node.target, ctx, seen)
  if (node.kind === 'union') {
    for (const member of node.members) {
      const description = descriptionFor(member, ctx, seen)
      if (description) return description
    }
  }
  return undefined
}

function extractParams(schema: Schema.Schema.AnyNoContext, ctx: RenderContext): ParamInfo[] {
  const transformDefaults = extractDefaultsFromTransformation(schema.ast)
  const inputNode = resolveObject(ctx.shape.root, ctx)
  if (!inputNode) return []

  return inputNode.fields.map((field) => {
    const desc = descriptionFor(field.value, ctx)
    const description = desc && !isNoiseDescription(desc)
      ? desc
      : field.meta.description && !isNoiseDescription(field.meta.description)
        ? field.meta.description
        : undefined
    const defaultValue = field.meta.defaultValue ?? transformDefaults.get(field.name)

    return {
      name: field.name,
      optional: field.optional,
      type: typeToString(field.value, field.optional, 1, ctx, new Set()),
      description,
      defaultValue,
    }
  })
}

function makeContext(schema: Schema.Schema.AnyNoContext, options?: RenderSchemaOptions): RenderContext {
  return {
    shape: inspectSchemaShape(schema, { maxNodes: options?.maxNodes }),
  }
}

export function renderSchemaType(
  schema: Schema.Schema.Any,
  options?: RenderSchemaOptions,
): string {
  const ctx = makeContext(schema as Schema.Schema.AnyNoContext, options)
  return typeToString(ctx.shape.root, false, 0, ctx, new Set())
}

export function renderSchemaParams(
  inputSchema: Schema.Schema.AnyNoContext,
  options?: RenderSchemaOptions,
): string {
  const ctx = makeContext(inputSchema, options)
  const params = extractParams(inputSchema, ctx)

  if (params.length === 0) {
    return 'Expected parameters: (none)'
  }

  const lines = params.map((param) => {
    const opt = param.optional ? '?' : ''
    let line = `  ${param.name}${opt}: ${param.type}`

    const commentParts: string[] = []
    if (param.description) commentParts.push(param.description)
    if (param.defaultValue !== undefined) commentParts.push(`default: ${formatDefaultValue(param.defaultValue)}`)

    if (commentParts.length > 0) {
      line += ` // ${commentParts.join(' - ')}`
    }

    return line
  })

  return `Expected parameters:\n${lines.join('\n')}`
}

function renderOneTool(tool: RenderableSchemaTool, options?: RenderSchemaOptions): string {
  const ctx = makeContext(tool.inputSchema, options)
  const params = extractParams(tool.inputSchema, ctx)
  const returnType = renderSchemaType(tool.outputSchema, options)
  const lines: string[] = []

  lines.push(`### ${tool.name}`)
  if (tool.description) {
    lines.push(tool.description)
  }
  lines.push('')

  const paramLines = params.map((param) => {
    const opt = param.optional ? '?' : ''
    const comment = buildComment(param.description, param.defaultValue)
    return `\t${param.name}${opt}: ${param.type}${comment}`
  })

  lines.push(`${tool.name}({`)
  lines.push(paramLines.join('\n'))
  lines.push(`}) -> ${returnType}`)

  return lines.join('\n')
}

export function renderToolDocs(
  tools: readonly RenderableSchemaTool[],
  options?: RenderSchemaOptions,
): string {
  return tools.map((tool) => renderOneTool(tool, options)).join('\n\n')
}
