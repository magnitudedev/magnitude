/**
 * Invoke, parameter, and filter frame handlers.
 */

import type { TurnEngineEvent, RegisteredTool, ToolParseErrorEvent, FilterReady, DeepPaths, StructuralParseError, ToolParseError } from '../../types'
import type { Op } from '../../machine'
import type { Frame, InvokeFrame, ParameterFrame, FilterFrame, FieldType } from '../types'
import { INVOKE_VALID_TAGS, PARAMETER_VALID_TAGS, FILTER_VALID_TAGS } from '../types'
import { coerceScalarValue } from '../coerce'
import { createStreamingJsonParser } from '../../jsonish/parser'
import { coerceToStreamingPartial } from '../../jsonish/coercer'
import { deriveParameters } from '../../engine/parameter-schema'
import { generateToolInterface } from '@magnitudedev/tools'

// =============================================================================
// InvokeContext — shared state passed to all invoke-related handlers
// =============================================================================

export interface InvokeContext {
  tools: ReadonlyMap<string, RegisteredTool>
  toolSchemas: Map<string, ReturnType<typeof deriveParameters>>
  endCurrentProse: () => void
  apply: (ops: Op<Frame, TurnEngineEvent>[]) => void
  emit: (event: TurnEngineEvent) => void
  emitStructuralError: (error: StructuralParseError) => void
  emitToolError: (error: ToolParseError, context: { toolCallId: string; tagName: string; toolName: string; group: string; correctToolShape?: string }) => void
  findFrame: <T extends Frame['type']>(type: T) => Extract<Frame, { type: T }> | undefined
  finalizeInvoke: (frame: InvokeFrame) => void
  onFilterReady?: (event: FilterReady) => void
  generateId: () => string
}

// =============================================================================
// Helpers
// =============================================================================

function getCorrectToolShape(toolTag: string, tools: ReadonlyMap<string, RegisteredTool>): string | undefined {
  const registered = tools.get(toolTag)
  if (!registered) return undefined
  try {
    const result = generateToolInterface(registered.tool, registered.groupName ?? 'tools', undefined, { extractCommon: false, showErrors: false })
    return result.signature
  } catch {
    return undefined
  }
}

function getFieldType(
  toolTag: string,
  paramName: string,
  toolSchemas: Map<string, ReturnType<typeof deriveParameters>>,
): FieldType {
  const schema = toolSchemas.get(toolTag)
  if (!schema) return 'unknown'
  const param = schema.parameters.get(paramName)
  if (!param) return 'unknown'
  if (param.type === 'json') return 'json'
  if (typeof param.type === 'object' && param.type._tag === 'enum') return 'string'
  return param.type as FieldType
}

// =============================================================================
// openInvoke
// =============================================================================

export function openInvoke(variant: string | undefined, ctx: InvokeContext): void {
  if (!variant) {
    ctx.emitStructuralError({ _tag: 'MissingToolName', detail: '<|invoke> requires a tool name, e.g. <|invoke:shell>' })
    return
  }

  const toolCallId = ctx.generateId()
  const toolTag = variant
  const parts = toolTag.split(':')
  const group = parts.length > 1 ? parts[0] : 'default'
  const toolName = parts.length > 1 ? parts.slice(1).join(':') : toolTag

  ctx.endCurrentProse()

  const registered = ctx.tools.get(toolTag)

  if (!registered) {
    ctx.apply([{
      type: 'push',
      frame: {
        type: 'invoke',
        toolCallId,
        toolTag,
        toolName,
        group,
        known: false,
        dead: true,
        hasFilter: false,
        fieldStates: new Map(),
        seenParams: new Set(),
        validTags: INVOKE_VALID_TAGS,
      },
    }])
    ctx.emitStructuralError(
      { _tag: 'UnknownTool', tagName: toolTag, detail: `Unknown tool tag: '${toolTag}'` },
    )
  } else {
    ctx.apply([
      {
        type: 'push',
        frame: {
          type: 'invoke',
          toolCallId,
          toolTag,
          toolName: registered.tool.name,
          group: registered.groupName,
          known: true,
          dead: false,
          hasFilter: false,
          fieldStates: new Map(),
          seenParams: new Set(),
          validTags: INVOKE_VALID_TAGS,
        },
      },
      {
        type: 'emit',
        event: {
          _tag: 'ToolInputStarted',
          toolCallId,
          tagName: toolTag,
          toolName: registered.tool.name,
          group: registered.groupName,
        },
      },
    ])
  }
}

// =============================================================================
// openFilter (piped close = filter start)
// =============================================================================

export function openFilter(invokeFrame: InvokeFrame, pipe: string, ctx: InvokeContext): void {
  ctx.apply([
    { type: 'replace', frame: { ...invokeFrame, hasFilter: true } },
    {
      type: 'push',
      frame: {
        type: 'filter',
        toolCallId: invokeFrame.toolCallId,
        filterType: pipe,
        query: '',
        validTags: FILTER_VALID_TAGS,
      },
    },
  ])
}

// =============================================================================
// closeFilter
// =============================================================================

export function closeFilter(filterFrame: FilterFrame, ctx: InvokeContext): void {
  ctx.apply([{ type: 'pop' }])
  if (ctx.onFilterReady) {
    ctx.onFilterReady({ _tag: 'FilterReady', toolCallId: filterFrame.toolCallId, query: filterFrame.query })
  }
  const invokeFrame = ctx.findFrame('invoke')
  if (invokeFrame) {
    ctx.finalizeInvoke(invokeFrame)
  }
}

// =============================================================================
// openParameter
// =============================================================================

export function openParameter(paramName: string, invokeFrame: InvokeFrame, ctx: InvokeContext): void {
  // Dead invoke — absorb parameter silently
  if (invokeFrame.dead) {
    ctx.apply([{
      type: 'push',
      frame: {
        type: 'parameter',
        toolCallId: invokeFrame.toolCallId,
        paramName,
        dead: true,
        rawValue: '',
        jsonishParser: null,
        fieldType: 'unknown',
        validTags: PARAMETER_VALID_TAGS,
      },
    }])
    return
  }

  const schema = ctx.toolSchemas.get(invokeFrame.toolTag)
  const knownParams = schema ? schema.parameters : null

  // Unknown parameter
  if (knownParams !== null && !knownParams.has(paramName)) {
    ctx.apply([{
      type: 'push',
      frame: {
        type: 'parameter',
        toolCallId: invokeFrame.toolCallId,
        paramName,
        dead: true,
        rawValue: '',
        jsonishParser: null,
        fieldType: 'unknown',
        validTags: PARAMETER_VALID_TAGS,
      },
    }])
    ctx.emitToolError(
      {
        _tag: 'UnknownParameter',
        toolCallId: invokeFrame.toolCallId,
        tagName: invokeFrame.toolTag,
        parameterName: paramName,
        detail: `Unknown parameter '${paramName}' for tool '${invokeFrame.toolTag}'`,
      },
      {
        toolCallId: invokeFrame.toolCallId,
        tagName: invokeFrame.toolTag,
        toolName: invokeFrame.toolName,
        group: invokeFrame.group,
        correctToolShape: getCorrectToolShape(invokeFrame.toolTag, ctx.tools),
      },
    )
    return
  }

  // Duplicate parameter — silently absorb
  if (invokeFrame.seenParams.has(paramName)) {
    ctx.apply([{
      type: 'push',
      frame: {
        type: 'parameter',
        toolCallId: invokeFrame.toolCallId,
        paramName,
        dead: true,
        rawValue: '',
        jsonishParser: null,
        fieldType: 'unknown',
        validTags: PARAMETER_VALID_TAGS,
      },
    }])
    return
  }

  invokeFrame.seenParams.add(paramName)

  const fieldType = getFieldType(invokeFrame.toolTag, paramName, ctx.toolSchemas)
  const jsonishParser = fieldType === 'json' ? createStreamingJsonParser() : null

  invokeFrame.fieldStates.set(paramName, {
    paramName,
    rawValue: '',
    coercedValue: undefined,
    errored: false,
    errorDetail: undefined,
    complete: false,
  })

  ctx.apply([{
    type: 'push',
    frame: {
      type: 'parameter',
      toolCallId: invokeFrame.toolCallId,
      paramName,
      dead: false,
      rawValue: '',
      jsonishParser,
      fieldType,
      validTags: PARAMETER_VALID_TAGS,
    },
  }])
}

// =============================================================================
// finalizeParameter
// =============================================================================

export function finalizeParameter(paramFrame: ParameterFrame, ctx: InvokeContext): void {
  if (paramFrame.dead) {
    ctx.apply([{ type: 'pop' }])
    return
  }

  if (paramFrame.jsonishParser !== null) {
    paramFrame.jsonishParser.end()
  }

  let coercedValue: unknown
  let errored = false
  let errorDetail: string | undefined

  const invokeFrame = ctx.findFrame('invoke')

  if (paramFrame.jsonishParser !== null) {
    const partial = paramFrame.jsonishParser.partial
    if (partial === undefined) {
      coercedValue = paramFrame.rawValue.trim() || undefined
    } else {
      const invokeTag = invokeFrame?.toolTag ?? ''
      const paramSchema = ctx.toolSchemas.get(invokeTag)?.parameters.get(paramFrame.paramName)
      if (paramSchema) {
        const registered = ctx.tools.get(invokeTag)
        if (registered) {
          try {
            const result = coerceToStreamingPartial(partial, registered.tool.inputSchema.ast)
            coercedValue = result?.value
          } catch {
            coercedValue = paramFrame.rawValue
            errored = true
            errorDetail = 'JSON coercion failed'
          }
        } else {
          coercedValue = paramFrame.rawValue
        }
      } else {
        coercedValue = paramFrame.rawValue
      }
    }
  } else {
    const { value, ok } = coerceScalarValue(paramFrame.rawValue, paramFrame.fieldType)
    coercedValue = value
    if (!ok) {
      errored = true
      errorDetail = `Cannot coerce '${paramFrame.rawValue}' to ${paramFrame.fieldType}`
    }
  }

  if (invokeFrame) {
    const fieldState = invokeFrame.fieldStates.get(paramFrame.paramName)
    if (fieldState) {
      fieldState.rawValue = paramFrame.rawValue
      fieldState.coercedValue = coercedValue
      fieldState.errored = errored
      fieldState.errorDetail = errorDetail
      fieldState.complete = true
    }
  }

  const path = [paramFrame.paramName] as unknown as DeepPaths<unknown>
  ctx.apply([
    {
      type: 'emit',
      event: {
        _tag: 'ToolInputFieldComplete',
        toolCallId: paramFrame.toolCallId,
        field: paramFrame.paramName as string & keyof unknown,
        path,
        value: coercedValue,
      },
    },
    { type: 'pop' },
  ])
}

// =============================================================================
// finalizeInvoke
// =============================================================================

export function finalizeInvoke(invokeFrame: InvokeFrame, ctx: InvokeContext): void {
  if (invokeFrame.dead) {
    ctx.apply([{ type: 'pop' }])
    return
  }

  const { toolCallId, toolTag, toolName, group } = invokeFrame
  const schema = ctx.toolSchemas.get(toolTag)
  const correctToolShape = getCorrectToolShape(toolTag, ctx.tools)
  const errors: ToolParseErrorEvent[] = []

  // Collect all errored fields
  for (const [, fieldState] of invokeFrame.fieldStates) {
    if (fieldState.errored) {
      errors.push({
        _tag: 'ToolParseError',
        toolCallId,
        tagName: toolTag,
        toolName,
        group,
        correctToolShape,
        error: {
          _tag: 'SchemaCoercionError',
          toolCallId,
          tagName: toolTag,
          parameterName: fieldState.paramName,
          detail: fieldState.errorDetail ?? 'Coercion failed',
        },
      })
    }
  }

  // Collect all missing required fields
  if (schema) {
    for (const [paramName, paramSchema] of schema.parameters) {
      if (paramSchema.required && !invokeFrame.fieldStates.has(paramName)) {
        errors.push({
          _tag: 'ToolParseError',
          toolCallId,
          tagName: toolTag,
          toolName,
          group,
          correctToolShape,
          error: {
            _tag: 'MissingRequiredField',
            toolCallId,
            tagName: toolTag,
            parameterName: paramName,
            detail: `Missing required field '${paramName}' for tool '${toolTag}'`,
          },
        })
      }
    }
  }

  if (errors.length > 0) {
    ctx.apply([{ type: 'pop' }])
    for (const err of errors) {
      ctx.emit(err)
    }
    return
  }

  // Happy path
  const input: Record<string, unknown> = {}
  for (const [paramName, fieldState] of invokeFrame.fieldStates) {
    input[paramName] = fieldState.coercedValue
  }

  ctx.apply([
    { type: 'emit', event: { _tag: 'ToolInputReady', toolCallId, input } },
    { type: 'pop' },
  ])
}
