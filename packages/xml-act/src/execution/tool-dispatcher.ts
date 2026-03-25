/**
 * ToolDispatcher — executes a parsed tool element.
 *
 * Triggered when TagClosed fires (element is complete).
 * Handles: input building, validation, interceptor, execution, and output-tree serialization.
 *
 * The dispatcher receives an `emit` callback from the reactor. It calls emit
 * at the right moments (ToolExecutionStarted before execution, ToolExecutionEnded
 * after). The reactor provides emitAndFold as the callback, so state stays in
 * sync automatically. The dispatcher never touches the queue or replay state directly.
 */

import { Effect, Either } from "effect"
import { AST } from "@effect/schema"
import { Schema } from "@effect/schema"
import type { ToolContext } from '@magnitudedev/tools'
import type { ParsedElement, TagParseErrorDetail } from '../format/types'
import type {
  RegisteredTool,
  XmlRuntimeEvent,
  XmlToolResult,
  ToolInterceptor,
  InterceptorContext,
} from '../types'
import { buildInput } from './input-builder'
import { buildOutputTree } from '../output-tree'

// =============================================================================
// Schema AST helpers
// =============================================================================

function unwrapAst(ast: AST.AST): AST.AST {
  if (ast._tag === 'Transformation') return unwrapAst(ast.from)
  if (ast._tag === 'Refinement') return unwrapAst(ast.from)
  return ast
}

function findMissingRequiredFields(
  rawInput: Record<string, unknown>,
  schemaAst: AST.AST,
): string[] {
  const missing: string[] = []
  const ast = unwrapAst(schemaAst)
  if (ast._tag !== 'TypeLiteral') return missing

  for (const prop of ast.propertySignatures) {
    const name = String(prop.name)
    if (!prop.isOptional && !(name in rawInput)) {
      missing.push(name)
    }
  }

  return missing
}

// =============================================================================
// Dispatch result (returned to reactor)
// =============================================================================

export type DispatchResult =
  | { readonly _tag: 'Dispatched' }
  | { readonly _tag: 'ParseError'; readonly error: TagParseErrorDetail }

export interface DispatchContext {
  readonly tools: ReadonlyMap<string, RegisteredTool>
  readonly interceptor: ToolInterceptor | undefined
  readonly emit: (event: XmlRuntimeEvent) => Effect.Effect<void>
  readonly toolContext?: ToolContext<unknown>
}

// =============================================================================
// Tool execution
// =============================================================================

function executeToolEffect(
  registered: RegisteredTool,
  input: unknown,
  toolContext?: ToolContext<unknown>,
): Effect.Effect<Either.Either<unknown, unknown>> {
  return Effect.suspend(() => {
    const exec = (registered.tool.execute as (i: unknown, ctx?: ToolContext<unknown>) => Effect.Effect<unknown, unknown, unknown>)(input, toolContext)

    if (registered.layerProvider) {
      return registered.layerProvider().pipe(
        Effect.flatMap((layer) =>
          exec.pipe(Effect.provide(layer))
        ),
        Effect.either,
      ) as Effect.Effect<Either.Either<unknown, unknown>>
    }

    return Effect.either(exec) as Effect.Effect<Either.Either<unknown, unknown>>
  })
}

/**
 * Dispatch a completed tool element for execution.
 *
 * Emits ToolExecutionStarted/ToolExecutionEnded via the emit callback
 * at the correct moments (started before execution, ended after).
 * Returns a DispatchResult so the reactor knows whether it was a parse error
 * (which the reactor handles itself) or a successful dispatch.
 */
export function dispatchTool(
  element: ParsedElement,
  ctx: DispatchContext,
): Effect.Effect<DispatchResult> {
  return Effect.gen(function* () {
    const registered = ctx.tools.get(element.tagName)

    if (!registered) {
      return { _tag: 'Dispatched' as const }
    }

    const { tool, groupName, binding, meta } = registered
    const observe = typeof element.attributes.get('observe') === 'string' ? String(element.attributes.get('observe')) : '.'
    const sanitizedAttributes = new Map(element.attributes)
    sanitizedAttributes.delete('observe')
    const sanitizedElement: ParsedElement = {
      ...element,
      attributes: sanitizedAttributes,
    }

    // 1. Build input from element + binding
    let rawInput = buildInput(sanitizedElement, binding)

    // 2. Check for missing required fields
    const missingFields = findMissingRequiredFields(rawInput, tool.inputSchema.ast)
    if (missingFields.length > 0) {
      const fieldList = missingFields.map(f => `'${f}'`).join(', ')
      return {
        _tag: 'ParseError' as const,
        error: {
          _tag: 'MissingRequiredFields' as const,
          id: element.toolCallId,
          tagName: element.tagName,
          fields: missingFields,
          detail: `Required field${missingFields.length > 1 ? 's' : ''} ${fieldList} missing on <${element.tagName}>`,
        },
      }
    }

    // 3. Validate against schema
    let input: unknown
    try {
      input = Schema.decodeUnknownSync(tool.inputSchema)(rawInput)
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      return {
        _tag: 'ParseError' as const,
        error: {
          _tag: 'MissingRequiredFields' as const,
          id: element.toolCallId,
          tagName: element.tagName,
          fields: [],
          detail: `Validation failed on <${element.tagName}>: ${msg}`,
        },
      }
    }

    // 4. Interceptor beforeExecute
    if (ctx.interceptor) {
      const interceptorCtx: InterceptorContext = {
        toolCallId: element.toolCallId, tagName: element.tagName,
        group: groupName, toolName: tool.name, input, meta,
      }
      const decision = yield* ctx.interceptor.beforeExecute(interceptorCtx)
      if (decision._tag === 'Reject') {
        // Emit both events even for rejections — they happened
        yield* ctx.emit({
          _tag: 'ToolExecutionStarted',
          toolCallId: element.toolCallId, group: groupName, toolName: tool.name,
          input, cached: false,
        })
        yield* ctx.emit({
          _tag: 'ToolExecutionEnded',
          toolCallId: element.toolCallId, group: groupName, toolName: tool.name,
          result: { _tag: 'Rejected', rejection: decision.rejection },
        })
        return { _tag: 'Dispatched' as const }
      }
      if (decision.modifiedInput !== undefined) {
        input = decision.modifiedInput
      }
    }

    // 5. Emit ToolExecutionStarted BEFORE execution
    yield* ctx.emit({
      _tag: 'ToolExecutionStarted',
      toolCallId: element.toolCallId, group: groupName, toolName: tool.name,
      input, cached: false,
    })

    // 6. Execute tool
    const executionResult = yield* executeToolEffect(registered, input, ctx.toolContext)

    let result: XmlToolResult

    if (Either.isLeft(executionResult)) {
      const e = executionResult.left
      const error = e instanceof Error
        ? e.message
        : (typeof e === 'object' && e !== null && 'message' in e && typeof (e as { message: unknown }).message === 'string')
          ? (e as { message: string }).message
          : String(e)
      result = { _tag: 'Error', error }
    } else {
      const output = executionResult.right
      
      const outputTree = buildOutputTree(element.tagName, output, tool.bindings.xmlOutput, undefined, { outputSchema: tool.outputSchema })
      result = { _tag: 'Success', output, outputTree: { tag: element.tagName, tree: outputTree }, query: observe }
    }

    // 7. Interceptor afterExecute
    if (ctx.interceptor?.afterExecute && result._tag === 'Success') {
      const interceptorCtx: InterceptorContext & { result: unknown } = {
        toolCallId: element.toolCallId, tagName: element.tagName,
        group: groupName, toolName: tool.name, input, meta,
        result: result.output,
      }
      const postDecision = yield* ctx.interceptor.afterExecute(interceptorCtx)
      if (postDecision._tag === 'Reject') {
        result = { _tag: 'Rejected', rejection: postDecision.rejection }
      }
    }

    // 8. Emit ToolExecutionEnded AFTER execution
    yield* ctx.emit({
      _tag: 'ToolExecutionEnded',
      toolCallId: element.toolCallId, group: groupName, toolName: tool.name,
      result,
    })

    return { _tag: 'Dispatched' as const }
  })
}
