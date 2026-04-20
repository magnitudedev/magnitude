/**
 * ParameterAccumulator — Streaming accumulator for tool input parameters.
 *
 * Implements StreamingAccumulatorLike<TInput, TurnEngineEvent>.
 * Receives TurnEngineEvent events and produces StreamingPartial via its `current` property.
 *
 * Works with the existing TurnEngineEvent flow:
 * - ToolInputStarted: initialize
 * - ToolInputFieldValue: scalar field streaming (text value accumulating)
 * - ToolInputReady: finalize all fields
 *
 * For JSON parameters, uses StreamingJsonParser + SchemaCoercer when
 * per-chunk events become available. Currently falls back to
 * direct JSON.parse on ToolInputReady.
 */

import { AST } from '@effect/schema'
import type { StreamingPartial, StreamingLeaf } from '@magnitudedev/tools'
import type { TurnEngineEvent } from '../types'
import type { ToolSchema, ParameterSchema, ScalarType } from '../execution/parameter-schema'
import { createStreamingJsonParser } from './parser'
import type { StreamingJsonParser, ParsedValue } from './types'
import { coerceToStreamingPartial } from './coercer'

// =============================================================================
// AST Helpers
// =============================================================================

function unwrapAst(ast: AST.AST): AST.AST {
  if (ast._tag === 'Transformation') return unwrapAst(ast.from)
  if (ast._tag === 'Refinement') return unwrapAst(ast.from)
  if (ast instanceof AST.PropertySignature) return unwrapAst(ast.type)
  return ast
}

/**
 * Navigate the schema AST to find the type for a dotted field path.
 */
function getFieldAst(rootAst: AST.AST, fieldPath: string): AST.AST {
  const parts = fieldPath.split('.')
  let currentAst = rootAst

  for (const part of parts) {
    const unwrapped = unwrapAst(currentAst)

    if (unwrapped._tag === 'TypeLiteral') {
      const prop = unwrapped.propertySignatures.find(p => String(p.name) === part)
      if (prop) {
        currentAst = prop.type
      } else {
        break
      }
    } else if (unwrapped._tag === 'Union') {
      const variantWithField = unwrapped.types.find(t => {
        const u = unwrapAst(t)
        return u._tag === 'TypeLiteral' && u.propertySignatures.some(p => String(p.name) === part)
      })
      if (variantWithField) {
        const u = unwrapAst(variantWithField)
        if (u._tag === 'TypeLiteral') {
          const prop = u.propertySignatures.find(p => String(p.name) === part)
          if (prop) {
            currentAst = prop.type
          }
        }
      } else {
        break
      }
    } else {
      break
    }
  }

  return currentAst
}

// =============================================================================
// Types
// =============================================================================

interface ScalarParamState {
  readonly _tag: 'scalar'
  readonly schema: ParameterSchema
  buffer: string
  complete: boolean
  hasData: boolean
}

interface JsonParamState {
  readonly _tag: 'json'
  readonly schema: ParameterSchema
  readonly fieldAst: AST.AST
  parser: StreamingJsonParser
  complete: boolean
  hasData: boolean
}

type ParamState = ScalarParamState | JsonParamState

interface AccumulatorState {
  params: Map<string, ParamState>
  active: boolean
  toolCallId: string | null
}

// =============================================================================
// Helpers
// =============================================================================

function createParamState(schema: ParameterSchema, schemaAst: AST.AST): ParamState {
  if (schema.type === 'json') {
    const fieldAst = getFieldAst(schemaAst, schema.name)
    return {
      _tag: 'json',
      schema,
      fieldAst,
      parser: createStreamingJsonParser(),
      complete: false,
      hasData: false,
    }
  }
  return {
    _tag: 'scalar',
    schema,
    buffer: '',
    complete: false,
    hasData: false,
  }
}

function coerceScalar(value: string, scalarType: ScalarType): unknown {
  const trimmed = value.trim()
  switch (scalarType) {
    case 'string': return trimmed
    case 'number': {
      const num = parseFloat(trimmed)
      return isNaN(num) ? trimmed : num
    }
    case 'boolean': {
      const lower = trimmed.toLowerCase()
      if (lower === 'true' || lower === '1') return true
      if (lower === 'false' || lower === '0') return false
      return trimmed
    }
    default: {
      if (typeof scalarType === 'object' && scalarType._tag === 'enum') {
        if (scalarType.values.includes(trimmed)) return trimmed
        const lowerTrimmed = trimmed.toLowerCase()
        for (const val of scalarType.values) {
          if (val.toLowerCase() === lowerTrimmed) return val
        }
        return trimmed
      }
      return trimmed
    }
  }
}

function buildScalarLeaf(state: ScalarParamState): StreamingLeaf<unknown> {
  if (state.complete) {
    return { isFinal: true, value: coerceScalar(state.buffer, state.schema.type as ScalarType) }
  }
  return { isFinal: false, value: state.buffer }
}

function parsedValueToStreamingPartial(parsed: ParsedValue): StreamingPartial<unknown> {
  switch (parsed._tag) {
    case 'string':
      return { isFinal: parsed.state === 'complete', value: parsed.value } as StreamingPartial<unknown>
    case 'number': {
      const num = parseFloat(parsed.value)
      return { isFinal: parsed.state === 'complete', value: isNaN(num) ? parsed.value : num } as StreamingPartial<unknown>
    }
    case 'boolean':
      return { isFinal: true, value: parsed.value } as StreamingPartial<unknown>
    case 'null':
      return { isFinal: true, value: null } as StreamingPartial<unknown>
    case 'object': {
      const result: Record<string, StreamingPartial<unknown>> = {}
      for (const [key, value] of parsed.entries) {
        result[key] = parsedValueToStreamingPartial(value)
      }
      return result as StreamingPartial<unknown>
    }
    case 'array':
      return parsed.items.map(item => parsedValueToStreamingPartial(item)) as StreamingPartial<unknown>
  }
}

function buildJsonPartial(state: JsonParamState): StreamingPartial<unknown> | undefined {
  const parsed = state.parser.partial
  if (parsed === undefined) return undefined
  return parsedValueToStreamingPartial(parsed)
}

function setNestedValue(obj: Record<string, unknown>, path: string, value: unknown): void {
  const parts = path.split('.')
  let current: Record<string, unknown> = obj
  for (let i = 0; i < parts.length - 1; i++) {
    const part = parts[i]
    if (!(part in current) || typeof current[part] !== 'object' || current[part] === null) {
      current[part] = {}
    }
    current = current[part] as Record<string, unknown>
  }
  current[parts[parts.length - 1]] = value
}

// =============================================================================
// ParameterAccumulator Implementation
// =============================================================================

/** @deprecated StreamingAccumulatorLike removed — accumulation moved to parser layer. */
export interface LegacyStreamingAccumulatorLike<TInput, TEvent = unknown> {
  ingest(event: TEvent): void;
  readonly current: StreamingPartial<TInput>;
  reset(): void;
}

export function createParameterAccumulator(
  toolSchema: ToolSchema,
  schemaAst: AST.AST
): { ingest(event: TurnEngineEvent): void; readonly current: StreamingPartial<unknown>; reset(): void; } {
  const state: AccumulatorState = {
    params: new Map(),
    active: false,
    toolCallId: null,
  }

  function initialize(): void {
    state.params.clear()
    for (const [name, paramSchema] of toolSchema.parameters) {
      state.params.set(name, createParamState(paramSchema, schemaAst))
    }
    state.active = true
  }

  function buildCurrent(): StreamingPartial<unknown> {
    const result: Record<string, unknown> = {}

    for (const [name, paramState] of state.params) {
      if (!paramState.hasData) continue

      let value: unknown
      switch (paramState._tag) {
        case 'scalar':
          value = buildScalarLeaf(paramState)
          break
        case 'json': {
          const jsonPartial = buildJsonPartial(paramState)
          if (jsonPartial !== undefined) {
            value = jsonPartial
          } else {
            continue
          }
          break
        }
      }

      if (name.includes('.')) {
        setNestedValue(result, name, value)
      } else {
        result[name] = value
      }
    }

    return result as StreamingPartial<unknown>
  }

  function ingest(event: TurnEngineEvent): void {
    switch (event._tag) {
      case 'ToolInputStarted': {
        state.toolCallId = event.toolCallId
        initialize()
        break
      }
      case 'ToolInputFieldComplete': {
        if (!state.active) break
        const field = String(event.field)
        const paramState = state.params.get(field)
        if (paramState === undefined) break

        paramState.hasData = true
        const textValue = String(event.value ?? '')

        switch (paramState._tag) {
          case 'scalar':
            paramState.buffer = textValue
            break
          case 'json':
            paramState.parser.push(textValue)
            break
        }
        break
      }
      case 'ToolInputReady': {
        if (!state.active) break
        // Mark all params as complete and finalize
        for (const [, paramState] of state.params) {
          paramState.complete = true
          if (paramState._tag === 'json') {
            paramState.parser.end()
          }
        }
        break
      }
      default:
        // Ignore other TurnEngineEvent types
        break
    }
  }

  function reset(): void {
    state.params.clear()
    state.active = false
    state.toolCallId = null
  }

  return {
    ingest,
    get current() { return buildCurrent() },
    reset,
  }
}
