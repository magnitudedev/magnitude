/**
 * Dispatcher — tool execution with interceptors.
 *
 * Receives pre-built input (from ToolInputReady, assembled by the parser/adapter).
 * Handles: schema validation, interceptor pipeline, execution, result construction.
 * Emits ToolExecutionStarted and ToolExecutionEnded with the typed ToolResult.
 * Rendering and persistence are separate concerns handled downstream.
 */

import { Effect, Either } from "effect"
import { Schema } from "@effect/schema"
import type { ToolContext } from '@magnitudedev/tools'
import type {
  RegisteredTool,
  TurnEngineEvent,
  ToolResult,
  ToolInterceptor,
  InterceptorContext,
} from './types'

// =============================================================================
// Types
// =============================================================================

export type DispatchResult =
  | { readonly _tag: 'Dispatched' }
  | { readonly _tag: 'DecodeFailure'; readonly detail: unknown }

export interface DispatchContext {
  readonly tools: ReadonlyMap<string, RegisteredTool>
  readonly interceptor: ToolInterceptor | undefined
  readonly emit: (event: TurnEngineEvent) => Effect.Effect<void>
  readonly toolContext?: ToolContext<unknown>
}

export interface DispatchInput {
  readonly toolName: string
  readonly toolCallId: string
  readonly input: unknown
}

// =============================================================================
// Tool execution
// =============================================================================

/**
 * Run a registered tool's execute and bracket it in an Either.
 *
 * Generic over `R`: the services the tool requires. The provided
 * `layerProvider` produces a `Layer<R, never, never>`; piping the execute
 * effect through `Effect.provide(layer)` resolves R to `never` cleanly.
 */
function executeToolEffect<R>(
  registered: RegisteredTool<R>,
  input: unknown,
  toolContext?: ToolContext<unknown>,
): Effect.Effect<Either.Either<unknown, unknown>> {
  return Effect.suspend(() => {
    const exec = (registered.tool.execute as (
      i: unknown,
      ctx?: ToolContext<unknown>,
    ) => Effect.Effect<unknown, unknown, R>)(input, toolContext)

    if (registered.layerProvider) {
      return registered.layerProvider().pipe(
        Effect.flatMap((layer) => exec.pipe(Effect.provide(layer))),
        Effect.either,
      )
    }

    return Effect.either(exec as Effect.Effect<unknown, unknown, never>)
  })
}

// =============================================================================
// Public API
// =============================================================================

/**
 * Dispatch a tool for execution.
 *
 * Emits ToolExecutionStarted and ToolExecutionEnded via the emit callback.
 * Returns DispatchResult so the engine knows the outcome.
 */
export function dispatchTool(
  request: DispatchInput,
  ctx: DispatchContext,
): Effect.Effect<DispatchResult> {
  return Effect.gen(function* () {
    const registered = ctx.tools.get(request.toolName)

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
      return {
        _tag: 'DecodeFailure' as const,
        detail: {
          kind: 'SchemaValidation',
          toolName: request.toolName,
          message: decodeResult.left.message,
        },
      }
    }

    let input: unknown = decodeResult.right

    // 2. Interceptor beforeExecute
    if (ctx.interceptor) {
      const interceptorCtx: InterceptorContext = {
        toolCallId: request.toolCallId,
        toolName: request.toolName,
        group: groupName,
        input,
        meta,
      }
      const decision = yield* ctx.interceptor.beforeExecute(interceptorCtx)
      if (decision._tag === 'Reject') {
        yield* ctx.emit({
          _tag: 'ToolExecutionStarted',
          toolCallId: request.toolCallId,
          toolName: request.toolName,
          group: groupName,
          input,
          cached: false,
        })
        yield* ctx.emit({
          _tag: 'ToolExecutionEnded',
          toolCallId: request.toolCallId,
          toolName: request.toolName,
          group: groupName,
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
      toolName: request.toolName,
      group: groupName,
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
      result = { _tag: 'Success', output: executionResult.right ?? null }
    }

    // 5. Interceptor afterExecute
    if (ctx.interceptor?.afterExecute && result._tag === 'Success') {
      const interceptorCtx: InterceptorContext & { result: unknown } = {
        toolCallId: request.toolCallId,
        toolName: request.toolName,
        group: groupName,
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
      toolName: request.toolName,
      group: groupName,
      result,
    })

    return { _tag: 'Dispatched' as const }
  })
}
