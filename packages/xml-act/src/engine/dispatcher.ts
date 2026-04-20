/**
 * Dispatcher — tool execution with interceptors and output observation.
 *
 * Receives pre-built input (from ToolInputReady, assembled by the parser).
 * Handles: schema validation, interceptor pipeline, execution, result construction,
 * and output observation (persist → query → render → emit ToolObservation).
 */

import { Effect, Either, Option } from "effect"
import { Schema } from "@effect/schema"
import type { ToolContext } from '@magnitudedev/tools'
import type {
  RegisteredTool,
  TurnEngineEvent,
  ToolResult,
  ToolInterceptor,
  InterceptorContext,
  ParseErrorDetail,
} from '../types'
import { queryOutput, renderFilteredResult, persistResult } from '../output'

// =============================================================================
// Types
// =============================================================================

export type DispatchResult =
  | { readonly _tag: 'Dispatched' }
  | { readonly _tag: 'ParseError'; readonly error: ParseErrorDetail }

export interface DispatchContext {
  readonly tools: ReadonlyMap<string, RegisteredTool>
  readonly interceptor: ToolInterceptor | undefined
  readonly emit: (event: TurnEngineEvent) => Effect.Effect<void>
  readonly toolContext?: ToolContext<unknown>
}

export interface DispatchInput {
  readonly tagName: string
  readonly toolCallId: string
  readonly input: unknown
  readonly filterQuery: string | null
  readonly turnId: string
  readonly resultsDir: string
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
        Effect.flatMap((layer) => exec.pipe(Effect.provide(layer))),
        Effect.either,
      ) as Effect.Effect<Either.Either<unknown, unknown>>
    }

    return Effect.either(exec) as Effect.Effect<Either.Either<unknown, unknown>>
  })
}

// =============================================================================
// Public API
// =============================================================================

/**
 * Dispatch a tool for execution.
 *
 * Emits ToolExecutionStarted, ToolExecutionEnded, and ToolObservation via the
 * emit callback. Returns DispatchResult so the engine knows the outcome.
 */
export function dispatchTool(
  request: DispatchInput,
  ctx: DispatchContext,
): Effect.Effect<DispatchResult> {
  return Effect.gen(function* () {
    const registered = ctx.tools.get(request.tagName)

    if (!registered) {
      return { _tag: 'Dispatched' as const }
    }

    const { tool, groupName, meta } = registered
    const rawInput = request.input

    // 1. Schema validation
    const decodeResult = yield* Effect.try({
      try: () => Schema.decodeUnknownSync(tool.inputSchema as Schema.Schema<unknown>)(rawInput),
      catch: (e) => e instanceof Error ? e : new Error(String(e)),
    }).pipe(Effect.either)

    if (Either.isLeft(decodeResult)) {
      const msg = decodeResult.left.message
      return {
        _tag: 'ParseError' as const,
        error: {
          _tag: 'SchemaCoercionError' as const,
          toolCallId: request.toolCallId,
          tagName: request.tagName,
          parameterName: '(input)',
          detail: `Schema validation failed for <${request.tagName}>: ${msg}`,
        } satisfies ParseErrorDetail,
      }
    }

    let input: unknown = decodeResult.right

    // 2. Interceptor beforeExecute
    if (ctx.interceptor) {
      const interceptorCtx: InterceptorContext = {
        toolCallId: request.toolCallId,
        tagName: request.tagName,
        group: groupName,
        toolName: tool.name,
        input,
        meta,
      }
      const decision = yield* ctx.interceptor.beforeExecute(interceptorCtx)
      if (decision._tag === 'Reject') {
        yield* ctx.emit({
          _tag: 'ToolExecutionStarted',
          toolCallId: request.toolCallId,
          tagName: request.tagName,
          group: groupName,
          toolName: tool.name,
          input,
          cached: false,
        })
        yield* ctx.emit({
          _tag: 'ToolExecutionEnded',
          toolCallId: request.toolCallId,
          tagName: request.tagName,
          group: groupName,
          toolName: tool.name,
          result: { _tag: 'Rejected', rejection: decision.rejection },
        })
        return { _tag: 'Dispatched' as const }
      }
      if (decision.modifiedInput !== undefined) {
        input = decision.modifiedInput
      }
    }

    // 3. Emit ToolExecutionStarted
    yield* ctx.emit({
      _tag: 'ToolExecutionStarted',
      toolCallId: request.toolCallId,
      tagName: request.tagName,
      group: groupName,
      toolName: tool.name,
      input,
      cached: false,
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
      result = { _tag: 'Success', output: executionResult.right, query: request.filterQuery }
    }

    // 5. Interceptor afterExecute
    if (ctx.interceptor?.afterExecute && result._tag === 'Success') {
      const interceptorCtx: InterceptorContext & { result: unknown } = {
        toolCallId: request.toolCallId,
        tagName: request.tagName,
        group: groupName,
        toolName: tool.name,
        input,
        meta,
        result: result.output,
      }
      const postDecision = yield* ctx.interceptor.afterExecute(interceptorCtx)
      if (postDecision._tag === 'Reject') {
        result = { _tag: 'Rejected', rejection: postDecision.rejection }
      }
    }

    // 6. Emit ToolExecutionEnded
    yield* ctx.emit({
      _tag: 'ToolExecutionEnded',
      toolCallId: request.toolCallId,
      tagName: request.tagName,
      group: groupName,
      toolName: tool.name,
      result,
    })

    // 7. Output observation (only on Success)
    if (result._tag === 'Success') {
      const output = result.output
      const query = request.filterQuery

      const resultPath = persistResult(output, request.turnId, request.toolCallId, request.resultsDir)
      const { filtered, isPartial } = queryOutput(output, query, resultPath)
      const contentParts = renderFilteredResult(tool.name, filtered, isPartial, resultPath)

      yield* ctx.emit({
        _tag: 'ToolObservation',
        toolCallId: request.toolCallId,
        tagName: request.tagName,
        query,
        content: contentParts,
      })
    }

    return { _tag: 'Dispatched' as const }
  })
}
