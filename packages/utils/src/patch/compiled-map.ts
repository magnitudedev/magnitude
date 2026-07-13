import { Schema, SchemaAST as AST, Option } from 'effect'
import type { CompiledField, CompiledMap, DecodedValue, DecodedSome, DecodedNone, JsonSubSchema, Path } from './types'
import type { JsonValue } from '../schema'
import type { JsonSafeSchema } from '../schema/json-safe'

// ---------------------------------------------------------------------------
// Type guards — proper narrowing, no `as`, no `unknown`
// ---------------------------------------------------------------------------

function isDecodedRecord(v: DecodedValue): v is { readonly [key: string]: DecodedValue } {
  return typeof v === 'object' && v !== null && !Array.isArray(v) && !isDecodedOption(v)
}

function isDecodedArray(v: DecodedValue): v is readonly DecodedValue[] {
  return Array.isArray(v)
}

/** Type guard for Option values within DecodedValue. */
function isDecodedOption(v: DecodedValue): v is DecodedSome | DecodedNone {
  return typeof v === 'object' && v !== null && !Array.isArray(v) && '_tag' in v && (v._tag === 'Some' || v._tag === 'None')
}

/** Unwrap Option to inner DecodedValue, or null if None. */
function unwrapOpt(v: DecodedValue): DecodedValue | null {
  if (isDecodedOption(v)) {
    return v._tag === 'Some' ? v.value : null
  }
  return v
}

/** Index into a decoded container at a key. */
function getChild(parent: DecodedValue, key: string | number): DecodedValue | null {
  const unwrapped = unwrapOpt(parent)
  if (unwrapped === null) return null
  if (isDecodedArray(unwrapped)) {
    const idx = typeof key === 'number' ? key : parseInt(String(key), 10)
    if (isNaN(idx) || idx < 0 || idx >= unwrapped.length) return null
    return unwrapped[idx]
  }
  if (isDecodedRecord(unwrapped)) {
    const val = unwrapped[String(key)]
    if (val === undefined) return null
    return val
  }
  return null
}

// ---------------------------------------------------------------------------
// AST peeling
// ---------------------------------------------------------------------------

function peelAll(ast: AST.AST): AST.AST {
  let current = ast
  for (;;) {
    if (AST.isSuspend(current)) {
      current = current.f()
    } else if (AST.isRefinement(current)) {
      current = current.from
    } else {
      return current
    }
  }
}

// ---------------------------------------------------------------------------
// Detect Option and default from a TypeLiteralTransformation's PST
// ---------------------------------------------------------------------------

interface PstInfo {
  isOption: boolean
  hasDefault: boolean
  defaultValue: JsonValue
}

function pstInfoForProperty(
  transformAst: AST.Transformation,
  key: string,
): PstInfo | null {
  if (!AST.isTypeLiteralTransformation(transformAst.transformation)) return null
  const pst = transformAst.transformation.propertySignatureTransformations.find(
    (p) => String(p.from) === key,
  )
  if (pst === undefined) return null

  const toAst = transformAst.to
  const toPs = AST.isTypeLiteral(toAst)
    ? toAst.propertySignatures.find((ps) => String(ps.name) === String(pst.to))
    : undefined

  const isOption = toPs !== undefined && AST.isDeclaration(toPs.type)

  let hasDefault = false
  let defaultValue: JsonValue = null
  if (!isOption) {
    const decResult = pst.decode(Option.none())
    if (Option.isSome(decResult)) {
      hasDefault = true
      const decoded = decResult.value
      if (decoded === null || typeof decoded === 'string' || typeof decoded === 'number' || typeof decoded === 'boolean') {
        defaultValue = decoded
      }
    }
  }

  return { isOption, hasDefault, defaultValue }
}

// ---------------------------------------------------------------------------
// Find discriminator key for a union — schema-driven
// ---------------------------------------------------------------------------

function findDiscriminator(unionAst: AST.Union): string | null {
  for (const member of unionAst.types) {
    const inner = peelAll(member)
    let tl: AST.TypeLiteral | null = null
    if (AST.isTypeLiteral(inner)) {
      tl = inner
    } else if (
      AST.isTransformation(inner) &&
      AST.isTypeLiteralTransformation(inner.transformation) &&
      AST.isTypeLiteral(inner.from)
    ) {
      tl = inner.from
    }
    if (tl === null) continue

    for (const ps of tl.propertySignatures) {
      const innerType = AST.isRefinement(ps.type) ? ps.type.from : ps.type
      if (AST.isLiteral(innerType) || AST.isUnion(innerType)) {
        return String(ps.name)
      }
    }
  }
  return null
}

// ---------------------------------------------------------------------------
// Match union member by discriminator value — schema-driven
// ---------------------------------------------------------------------------

function matchUnionMember(
  unionAst: AST.Union,
  discKey: string,
  discVal: JsonValue,
): AST.AST | null {
  for (const member of unionAst.types) {
    const inner = peelAll(member)
    let tl: AST.TypeLiteral | null = null
    if (AST.isTypeLiteral(inner)) {
      tl = inner
    } else if (
      AST.isTransformation(inner) &&
      AST.isTypeLiteralTransformation(inner.transformation) &&
      AST.isTypeLiteral(inner.from)
    ) {
      tl = inner.from
    }
    if (tl === null) continue

    for (const ps of tl.propertySignatures) {
      if (String(ps.name) !== discKey) continue
      const innerType = AST.isRefinement(ps.type) ? ps.type.from : ps.type

      if (AST.isLiteral(innerType) && discVal === innerType.literal) {
        return member
      }
      if (AST.isUnion(innerType)) {
        for (const litMember of innerType.types) {
          if (AST.isLiteral(litMember) && discVal === litMember.literal) {
            return member
          }
        }
      }
    }
  }
  return null
}

// ---------------------------------------------------------------------------
// Resolve a single path segment against an AST node
// ---------------------------------------------------------------------------

interface ResolveResult {
  childAst: AST.AST
  parentAst: AST.AST
  pstInfo: PstInfo | null
  innerAst: AST.AST | null
}

function resolveSegment(
  ast: AST.AST,
  key: string | number,
  decodedValue: DecodedValue,
): ResolveResult | null {
  const peeled = peelAll(ast)

  switch (peeled._tag) {
    case 'TypeLiteral': {
      for (const ps of peeled.propertySignatures) {
        if (String(ps.name) === String(key)) {
          return { childAst: ps.type, parentAst: ast, pstInfo: null, innerAst: null }
        }
      }
      for (const idx of peeled.indexSignatures) {
        if (AST.isStringKeyword(idx.parameter)) {
          return { childAst: idx.type, parentAst: ast, pstInfo: null, innerAst: null }
        }
      }
      return null
    }

    case 'Transformation': {
      if (!AST.isTypeLiteralTransformation(peeled.transformation)) return null
      const innerTl = peeled.from
      if (!AST.isTypeLiteral(innerTl)) return null

      for (const ps of innerTl.propertySignatures) {
        if (String(ps.name) === String(key)) {
          const pstInfo = pstInfoForProperty(peeled, String(key))
          let innerAst: AST.AST | null = null
          if (pstInfo?.isOption) {
            innerAst = ps.type
          }
          return { childAst: ps.type, parentAst: ast, pstInfo, innerAst }
        }
      }
      for (const idx of innerTl.indexSignatures) {
        if (AST.isStringKeyword(idx.parameter)) {
          return { childAst: idx.type, parentAst: ast, pstInfo: null, innerAst: null }
        }
      }
      return null
    }

    case 'TupleType': {
      const index = typeof key === 'number' ? key : parseInt(String(key), 10)
      if (isNaN(index) || index < 0) return null
      if (index < peeled.elements.length) {
        return { childAst: peeled.elements[index].type, parentAst: ast, pstInfo: null, innerAst: null }
      }
      if (peeled.rest.length > 0) {
        return { childAst: peeled.rest[0].type, parentAst: ast, pstInfo: null, innerAst: null }
      }
      return null
    }

    case 'Union': {
      const discKey = findDiscriminator(peeled)
      if (discKey === null) return null

      const discChild = getChild(decodedValue, discKey)
      if (discChild === null) return null

      const discUnwrapped = unwrapOpt(discChild)
      if (discUnwrapped === null) return null
      if (typeof discUnwrapped !== 'string' && typeof discUnwrapped !== 'number' && typeof discUnwrapped !== 'boolean') return null

      const memberAst = matchUnionMember(peeled, discKey, discUnwrapped)
      if (memberAst === null) return null

      const childVal = getChild(decodedValue, key)
      if (childVal === null) return null
      return resolveSegment(memberAst, key, childVal)
    }

    case 'Declaration': {
      const enc = AST.encodedAST(peeled)
      return resolveSegment(enc, key, decodedValue)
    }

    default:
      return null
  }
}

// ---------------------------------------------------------------------------
// Schema construction
// ---------------------------------------------------------------------------

function makeSchema(ast: AST.AST): JsonSubSchema {
  return Schema.make<DecodedValue, JsonValue, never>(ast)
}

// ---------------------------------------------------------------------------
// Compiled map implementation — generic over A (the decoded type)
// ---------------------------------------------------------------------------

export function compilePatchMap<S extends Schema.Schema.AnyNoContext>(
  schema: JsonSafeSchema<S>,
): CompiledMap<Schema.Schema.Type<S>> {
  const rootAst = schema.ast

  function resolvePath(path: Path, decodedRoot?: Schema.Schema.Type<S>): {
    childAst: AST.AST
    parentAst: AST.AST
    pstInfo: PstInfo | null
    innerAst: AST.AST | null
  } | null {
    if (path.length === 0) return null

    let currentAst = rootAst
    let currentDecoded: DecodedValue | undefined = undefined

    if (decodedRoot !== undefined) {
      currentDecoded = decodedRoot
    }

    for (let i = 0; i < path.length - 1; i++) {
      const key = path[i]

      const result = resolveSegment(currentAst, key, currentDecoded ?? Option.none())
      if (result === null) return null
      currentAst = result.childAst

      if (currentDecoded !== undefined) {
        currentDecoded = getChild(currentDecoded, key) ?? undefined
      }
    }

    const lastKey = path[path.length - 1]
    const final = resolveSegment(currentAst, lastKey, currentDecoded ?? Option.none())
    if (final === null) return null
    return final
  }

  function fieldAt(path: Path, decodedRoot?: Schema.Schema.Type<S>): CompiledField | null {
    const resolved = resolvePath(path, decodedRoot)
    if (resolved === null) return null

    const { childAst, pstInfo, innerAst } = resolved
    const isOption = pstInfo?.isOption ?? false
    const hasDefault = pstInfo?.hasDefault ?? false
    const defaultValue: JsonValue = pstInfo?.defaultValue ?? null

    return {
      isOption,
      hasDefault,
      defaultValue,
      subSchema: makeSchema(childAst),
      innerSubSchema: isOption && innerAst !== null ? makeSchema(innerAst) : null,
    }
  }

  function subSchemaAt(path: Path, decodedRoot?: Schema.Schema.Type<S>): JsonSubSchema | null {
    const resolved = resolvePath(path, decodedRoot)
    if (resolved === null) return null
    return makeSchema(resolved.childAst)
  }

  return { fieldAt, subSchemaAt }
}
