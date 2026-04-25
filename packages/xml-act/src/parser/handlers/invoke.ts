/**
 * Invoke, parameter, and filter frame handlers.
 *
 * All handlers are stateless objects implementing OpenHandler<TParent, TChild>
 * or CloseHandler<TFrame>. Parent frames are passed at call time via bindOpen/bindClose.
 * All effects are returned as ParserOp[] — no direct emit/apply calls.
 *
 * invokeOpenHandler:     OpenHandler<ProseFrame, InvokeFrame>
 * invokeCloseHandler:    CloseHandler<InvokeFrame>
 * parameterOpenHandler:  OpenHandler<InvokeFrame, ParameterFrame>
 * parameterCloseHandler: CloseHandler<ParameterFrame>
 * filterOpenHandler:     OpenHandler<InvokeFrame, FilterFrame>
 * filterCloseHandler:    CloseHandler<FilterFrame>
 */

import type {
  TurnEngineEvent,
  RegisteredTool,
  ToolParseErrorEvent,
  FilterReady,
  DeepPaths,
} from '../../types'
import type { Op } from '../../machine'
import type {
  Frame,
  ProseFrame,
  InvokeFrame,
  ParameterFrame,
  FilterFrame,
  FieldType,
} from '../types'
import type { OpenHandler, CloseHandler } from '../handler'
import type { HandlerContext, InvokeContext } from '../handler-context'
import { emitEvent, emitStructuralError, emitToolError, type ParserOp } from '../ops'
import { coerceScalarValue } from '../coerce'
import { createStreamingJsonParser } from '../../jsonish/parser'
import { coerceToStreamingPartial } from '../../jsonish/coercer'
import { deriveParameters } from '../../engine/parameter-schema'
import { generateToolInterface } from '@magnitudedev/tools'

// Re-export InvokeContext so index.ts can import it from here
export type { InvokeContext }

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
  toolSchemas: ReadonlyMap<string, ReturnType<typeof deriveParameters>>,
): FieldType {
  const schema = toolSchemas.get(toolTag)
  if (!schema) return 'unknown'
  const param = schema.parameters.get(paramName)
  if (!param) return 'unknown'
  if (param.type === 'json_object' || param.type === 'json_array') return 'json'
  if (typeof param.type === 'object' && param.type._tag === 'enum') return 'string'
  return param.type as FieldType
}

function makeDeadParamFrame(invokeFrame: InvokeFrame, paramName: string): ParameterFrame {
  return {
    type: 'parameter',
    openSpan: invokeFrame.openSpan,
    toolCallId: invokeFrame.toolCallId,
    paramName,
    dead: true,
    rawValue: '',
    jsonishParser: null,
    fieldType: 'unknown',
    invokeFrame,
  }
}

// =============================================================================
// finalizeInvokeOps — shared by invokeCloseHandler and filterCloseHandler
// =============================================================================

export function finalizeInvokeOps(invokeFrame: InvokeFrame, invokeCtx: InvokeContext): ParserOp[] {
  if (invokeFrame.dead) {
    return [{ type: 'pop' }]
  }

  const { toolCallId, toolTag, toolName, group } = invokeFrame
  const schema = invokeCtx.toolSchemas.get(toolTag)
  const correctToolShape = getCorrectToolShape(toolTag, invokeCtx.tools)
  const errorOps: ParserOp[] = []

  // Collect all errored fields
  for (const [, fieldState] of invokeFrame.fieldStates) {
    if (fieldState.errored) {
      errorOps.push(emitToolError(
        {
          _tag: 'SchemaCoercionError',
          toolCallId,
          tagName: toolTag,
          parameterName: fieldState.paramName,
          detail: fieldState.errorDetail ?? 'Coercion failed',
          primarySpan: fieldState.openSpan,
          relatedSpans: [invokeFrame.openSpan],
        },
        { toolCallId, tagName: toolTag, toolName, group, correctToolShape },
      ))
    }
  }

  // Collect all missing required fields
  if (schema) {
    for (const [paramName, paramSchema] of schema.parameters) {
      if (paramSchema.required && !invokeFrame.fieldStates.has(paramName)) {
        errorOps.push(emitToolError(
          {
            _tag: 'MissingRequiredField',
            toolCallId,
            tagName: toolTag,
            parameterName: paramName,
            detail: `Missing required field '${paramName}' for tool '${toolTag}'`,
            primarySpan: invokeFrame.openSpan,
          },
          { toolCallId, tagName: toolTag, toolName, group, correctToolShape },
        ))
      }
    }
  }

  if (errorOps.length > 0) {
    return [{ type: 'pop' }, ...errorOps]
  }

  // Happy path
  const input: Record<string, unknown> = {}
  for (const [paramName, fieldState] of invokeFrame.fieldStates) {
    input[paramName] = fieldState.coercedValue
  }

  return [
    emitEvent({ _tag: 'ToolInputReady', toolCallId, input }),
    { type: 'pop' },
  ]
}

// =============================================================================
// finalizeParameterOps — used by parameterCloseHandler and flush
// =============================================================================

export function finalizeParameterOps(paramFrame: ParameterFrame, invokeCtx: InvokeContext): ParserOp[] {
  if (paramFrame.dead) {
    return [{ type: 'pop' }]
  }

  if (paramFrame.jsonishParser !== null) {
    paramFrame.jsonishParser.end()
  }

  let coercedValue: unknown
  let errored = false
  let errorDetail: string | undefined

  const invokeFrame = paramFrame.invokeFrame  // stored at open time — no findFrame needed

  if (paramFrame.jsonishParser !== null) {
    const partial = paramFrame.jsonishParser.partial
    if (partial === undefined) {
      coercedValue = paramFrame.rawValue.trim() || undefined
    } else {
      const invokeTag = invokeFrame.toolTag
      const paramSchema = invokeCtx.toolSchemas.get(invokeTag)?.parameters.get(paramFrame.paramName)
      if (paramSchema) {
        const registered = invokeCtx.tools.get(invokeTag)
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

  // Update fieldState on the parent InvokeFrame
  const fieldState = invokeFrame.fieldStates.get(paramFrame.paramName)
  if (fieldState) {
    fieldState.rawValue = paramFrame.rawValue
    fieldState.coercedValue = coercedValue
    fieldState.errored = errored
    fieldState.errorDetail = errorDetail
    fieldState.complete = true
  }

  const path = [paramFrame.paramName] as unknown as DeepPaths<unknown>
  return [
    emitEvent({
      _tag: 'ToolInputFieldComplete',
      toolCallId: paramFrame.toolCallId,
      field: paramFrame.paramName as string & keyof unknown,
      path,
      value: coercedValue,
    }),
    { type: 'pop' },
  ]
}

// =============================================================================
// invokeOpenHandler / invokeCloseHandler
// =============================================================================

export const invokeOpenHandler: OpenHandler<ProseFrame, InvokeFrame> = {
  open(attrs, _parent, ctx, tokenSpan) {
    const toolTag = attrs.get('tool')
    if (!toolTag) {
      return [emitStructuralError({ _tag: 'MissingToolName', detail: '<magnitude:invoke> requires a tool attribute, e.g. <magnitude:invoke tool="shell">', primarySpan: tokenSpan })]
    }

    const toolCallId = ctx.generateId()
    const parts = toolTag.split(':')
    const group = parts.length > 1 ? parts[0] : 'default'
    const toolName = parts.length > 1 ? parts.slice(1).join(':') : toolTag

    const registered = ctx.invokeCtx.tools.get(toolTag)

    if (!registered) {
      return [
        {
          type: 'push',
          frame: {
            type: 'invoke',
            openSpan: tokenSpan,
            toolCallId,
            toolTag,
            toolName,
            group,
            known: false,
            dead: true,
            hasFilter: false,
            fieldStates: new Map(),
            seenParams: new Set(),
          },
        },
        emitStructuralError({ _tag: 'UnknownTool', tagName: toolTag, detail: `Unknown tool tag: '${toolTag}'`, primarySpan: tokenSpan }),
      ]
    }

    return [
      {
        type: 'push',
        frame: {
          type: 'invoke',
          openSpan: tokenSpan,
          toolCallId,
          toolTag,
          toolName: registered.tool.name,
          group: registered.groupName,
          known: true,
          dead: false,
          hasFilter: false,
          fieldStates: new Map(),
          seenParams: new Set(),
        },
      },
      emitEvent({
        _tag: 'ToolInputStarted',
        toolCallId,
        tagName: toolTag,
        toolName: registered.tool.name,
        group: registered.groupName,
        openSpan: tokenSpan,
      }),
    ]
  },
}

export const invokeCloseHandler: CloseHandler<InvokeFrame> = {
  close(top, ctx, _tokenSpan) {
    return finalizeInvokeOps(top, ctx.invokeCtx)
  },
}

// =============================================================================
// parameterOpenHandler / parameterCloseHandler
// =============================================================================

export const parameterOpenHandler: OpenHandler<InvokeFrame, ParameterFrame> = {
  open(attrs, parent, ctx, tokenSpan) {
    // parent is InvokeFrame — TypeScript enforces this at handler definition
    const paramName = attrs.get('name') ?? ''

    if (parent.dead) {
      return [{ type: 'push', frame: makeDeadParamFrame(parent, paramName) }]
    }

    const schema = ctx.invokeCtx.toolSchemas.get(parent.toolTag)
    const knownParams = schema ? schema.parameters : null

    if (knownParams !== null && !knownParams.has(paramName)) {
      return [
        { type: 'push', frame: makeDeadParamFrame(parent, paramName) },
        emitToolError(
          {
            _tag: 'UnknownParameter',
            toolCallId: parent.toolCallId,
            primarySpan: tokenSpan,
            relatedSpans: [parent.openSpan],
            tagName: parent.toolTag,
            parameterName: paramName,
            detail: `Unknown parameter '${paramName}' for tool '${parent.toolTag}'`,
          },
          {
            toolCallId: parent.toolCallId,
            tagName: parent.toolTag,
            toolName: parent.toolName,
            group: parent.group,
            correctToolShape: getCorrectToolShape(parent.toolTag, ctx.invokeCtx.tools),
          },
        ),
      ]
    }

    const isDuplicate = parent.seenParams.has(paramName)

    // Mutate seenParams and fieldStates on the InvokeFrame.
    // These Maps/Sets are mutable by design — accumulated during streaming.
    if (!isDuplicate) {
      parent.seenParams.add(paramName)
    }
    const fieldType = getFieldType(parent.toolTag, paramName, ctx.invokeCtx.toolSchemas)
    const jsonishParser = fieldType === 'json' ? createStreamingJsonParser() : null

    if (isDuplicate) {
      // Duplicate parameter — fail the tool call
      return [
        emitToolError(
          {
            _tag: 'DuplicateParameter',
            toolCallId: parent.toolCallId,
            primarySpan: tokenSpan,
            relatedSpans: [parent.openSpan],
            tagName: parent.toolTag,
            parameterName: paramName,
            detail: `Duplicate parameter '${paramName}' for tool '${parent.toolTag}'.`,
          },
          {
            toolCallId: parent.toolCallId,
            tagName: parent.toolTag,
            toolName: parent.toolTag,
            group: 'fs',
            correctToolShape: getCorrectToolShape(parent.toolTag, ctx.invokeCtx.tools),
          },
        ),
        { type: 'push', frame: makeDeadParamFrame(parent, paramName) },
      ]
    }

    parent.fieldStates.set(paramName, {
      paramName,
      openSpan: tokenSpan,
      rawValue: '',
      coercedValue: undefined,
      errored: false,
      errorDetail: undefined,
      complete: false,
    })

    return [{
      type: 'push',
      frame: {
        type: 'parameter',
        openSpan: tokenSpan,
        toolCallId: parent.toolCallId,
        paramName,
        dead: false,
        rawValue: '',
        jsonishParser,
        fieldType,
        invokeFrame: parent,  // stored reference — eliminates findFrame in finalizeParameterOps
      },
    }]
  },
}

export const parameterCloseHandler: CloseHandler<ParameterFrame> = {
  close(top, ctx, _tokenSpan) {
    return finalizeParameterOps(top, ctx.invokeCtx)
  },
}

// =============================================================================
// filterOpenHandler / filterCloseHandler
// =============================================================================

export const filterOpenHandler: OpenHandler<InvokeFrame, FilterFrame> = {
  open(attrs, parent, _ctx, tokenSpan) {
    const filterType = attrs.get('type') ?? 'jsonpath'
    return [
      { type: 'replace', frame: { ...parent, hasFilter: true } },
      {
        type: 'push',
        frame: {
          type: 'filter',
          openSpan: tokenSpan,
          toolCallId: parent.toolCallId,
          filterType,
          query: '',
          invokeFrame: parent,  // stored reference — eliminates findFrame in filterCloseHandler
        },
      },
    ]
  },
}

export const filterCloseHandler: CloseHandler<FilterFrame> = {
  close(top, ctx, _tokenSpan) {
    if (ctx.invokeCtx.onFilterReady) {
      ctx.invokeCtx.onFilterReady({ _tag: 'FilterReady', toolCallId: top.toolCallId, query: top.query })
    }
    return [{ type: 'pop' }]
  },
}
