/**
 * ToolDispatcher — executes a tool with validated input.
 *
 * The runtime has already built the input from parameters. The dispatcher
 * handles: schema validation, interceptor, execution, and result construction.
 *
 * The dispatcher receives an `emit` callback from the runtime. It calls emit
 * at the right moments (ToolExecutionStarted before execution, ToolExecutionEnded
 * after).
 */

import { Effect, Either } from "effect"
import { Schema } from "@effect/schema"
import type { ToolContext } from '@magnitudedev/tools'
import type {
  RegisteredTool,
  RuntimeEvent,
  ToolResult,
  ToolInterceptor,
  InterceptorContext,
  ParseErrorDetail,
} from '../types'

// =============================================================================
// Dispatch result (returned to runtime)
// =============================================================================

export type DispatchResult =
  | { readonly _tag: 'Dispatched' }
  | { readonly _tag: 'ParseError'; readonly error: ParseErrorDetail }

export interface DispatchContext {
  readonly tools: ReadonlyMap<string, RegisteredTool>
  readonly interceptor: ToolInterceptor | undefined
  readonly emit: (event: RuntimeEvent) => Effect.Effect<void>
  readonly toolContext?: ToolContext<unknown>
}

// =============================================================================
// Tool execution
// =============================================================================

function executeToolEffect(
  registered: RegisteredTool,
  input: unknown,
  toolContext?: ToolContext<unknown>,
): Effect.Effect<Either.Either<unknown, unknown>, never, never> {
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

// =============================================================================
// Public API
// =============================================================================

export interface ToolDispatchRequest {
  readonly tagName: string
  readonly toolCallId: string
  readonly input: Record<string, unknown>
}

/**
 * Dispatch a tool for execution.
 *
 * Emits ToolExecutionStarted/ToolExecutionEnded via the emit callback.
 * Returns a DispatchResult so the runtime knows whether it was a parse error
 * or a successful dispatch.
 */
export function dispatchTool(
  request: ToolDispatchRequest,
  ctx: DispatchContext,
): Effect.Effect<DispatchResult, never, never> {
  return Effect.gen(function* () {
    const registered = ctx.tools.get(request.tagName)

    if (!registered) {
      return { _tag: 'Dispatched' as const }
    }

    const { tool, groupName, meta } = registered
    const rawInput = request.input

    // 1. Validate against schema
    let input: unknown
    try {
      input = Schema.decodeUnknownSync(tool.inputSchema as Schema.Schema<unknown>)(rawInput)
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      return {
        _tag: 'ParseError' as const,
        error: {
          _tag: 'ToolValidationFailed' as const,
          id: request.toolCallId,
          tagName: request.tagName,
          detail: `Schema validation failed for <${request.tagName}>: ${msg}`,
        } satisfies ParseErrorDetail,
      }
    }

    // 2. Interceptor beforeExecute
    if (ctx.interceptor) {
      const interceptorCtx: InterceptorContext = {
        toolCallId: request.toolCallId, tagName: request.tagName,
        group: groupName, toolName: tool.name, input, meta,
      }
      const decision = yield* ctx.interceptor.beforeExecute(interceptorCtx)
      if (decision._tag === 'Reject') {
        yield* ctx.emit({
          _tag: 'ToolExecutionStarted',
          toolCallId: request.toolCallId, group: groupName, toolName: tool.name,
          input, cached: false,
        })
        yield* ctx.emit({
          _tag: 'ToolExecutionEnded',
          toolCallId: request.toolCallId, group: groupName, toolName: tool.name,
          result: { _tag: 'Rejected', rejection: decision.rejection },
        })
        return { _tag: 'Dispatched' as const }
      }
      if (decision.modifiedInput !== undefined) {
        input = decision.modifiedInput
      }
    }

    // 3. Emit ToolExecutionStarted BEFORE execution
    yield* ctx.emit({
      _tag: 'ToolExecutionStarted',
      toolCallId: request.toolCallId, group: groupName, toolName: tool.name,
      input, cached: false,
    })

    // 4. Execute tool
    const executionResult = yield* executeToolEffect(registered, input, ctx.toolContext)

    let result: ToolResult

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
      result = { _tag: 'Success', output, query: null }
    }

    // 5. Interceptor afterExecute
    if (ctx.interceptor?.afterExecute && result._tag === 'Success') {
      const interceptorCtx: InterceptorContext & { result: unknown } = {
        toolCallId: request.toolCallId, tagName: request.tagName,
        group: groupName, toolName: tool.name, input, meta,
        result: result.output,
      }
      const postDecision = yield* ctx.interceptor.afterExecute(interceptorCtx)
      if (postDecision._tag === 'Reject') {
        result = { _tag: 'Rejected', rejection: postDecision.rejection }
      }
    }

    // 6. Emit ToolExecutionEnded AFTER execution
    yield* ctx.emit({
      _tag: 'ToolExecutionEnded',
      toolCallId: request.toolCallId, group: groupName, toolName: tool.name,
      result,
    })

    return { _tag: 'Dispatched' as const }
  })
}
