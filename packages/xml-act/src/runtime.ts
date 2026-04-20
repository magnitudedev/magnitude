/**
 * Runtime — queue-based runtime that wires tokenizer → parser → reactor → stream.
 *
 * The reactor is internal. The only public API is createRuntime → Runtime.
 * Events are delivered via an Effect Queue for real-time streaming (no batching).
 *
 * For replay, consumers provide initialState with pre-populated toolOutcomes.
 * The reactor skips completed tools entirely and re-runs only the in-flight tool.
 */

import { Effect, Stream, Ref, Option, Queue, Cause } from "effect"
import type { ToolContext } from '@magnitudedev/tools'

import { createTokenizer } from './tokenizer'
import { createParser, type ParserEvent, type StructuralEvent } from './parser'
import { createShortId } from './util'
import type {
  Token,
  ParseEvent,
  ParameterStarted,
  ParameterChunk,
  ParameterComplete,
  FilterStarted,
  FilterChunk,
  FilterComplete,
  InvokeStarted,
  InvokeComplete,
  RuntimeEvent,
  ToolInterceptor,
  RegisteredTool,
  ReactorState,
  ParseErrorDetail,
} from './types'
import { TurnEngineCrash, ToolInterceptorTag } from './types'
import { dispatchTool, type DispatchContext, type DispatchResult } from './execution/tool-dispatcher'
import { buildInput, type ParsedInvoke, type ParsedParameter } from './execution/input-builder'
import { queryOutput, renderFilteredResult } from './output-query'
import { deriveParameters, type ToolSchema } from './execution/parameter-schema'
import { initialReactorState, foldReactorState } from './execution/reactor-state'
import { persistResult, getResultPath } from './result-persistence'

// =============================================================================
// Sentinel for end-of-stream
// =============================================================================

const END = Symbol('END')
type QueueItem = RuntimeEvent | typeof END

function describeDefect(defect: unknown): string {
  if (defect instanceof Error) return defect.message
  if (typeof defect === 'string') return defect
  if (typeof defect === 'object' && defect !== null) {
    const parts: string[] = []
    if ('_tag' in defect) parts.push(`[${(defect as any)._tag}]`)
    if ('name' in defect) parts.push(`${(defect as any).name}`)
    if ('message' in defect) parts.push(`${(defect as any).message}`)
    if (parts.length > 0) return parts.join(' ')
    try { return JSON.stringify(defect) } catch {}
  }
  return String(defect)
}

// =============================================================================
// Public API
// =============================================================================

export interface Runtime {
  readonly streamWith: <E extends { message: string } = Error>(
    textStream: Stream.Stream<string, E>,
    opts?: { readonly initialState?: ReactorState; readonly turnId?: string },
  ) => Stream.Stream<RuntimeEvent, TurnEngineCrash>
}

export type RuntimeConfig = import('./types').RuntimeConfig

/**
 * Create a runtime from configuration.
 * Derives parameter schemas from tool input schemas eagerly.
 */
export function createRuntime(config: RuntimeConfig): Runtime {
  // Derive parameter schemas from tool input schemas
  const toolSchemas = new Map<string, ToolSchema>()
  for (const [tagName, reg] of config.tools) {
    const schema = deriveParameters(reg.tool.inputSchema.ast)
    toolSchemas.set(tagName, schema)
  }

  return {
    streamWith<E extends { message: string } = Error>(
      textStream: Stream.Stream<string, E>,
      opts?: { readonly initialState?: ReactorState; readonly turnId?: string },
    ): Stream.Stream<RuntimeEvent, TurnEngineCrash> {
      return Stream.unwrapScoped(
        Effect.gen(function* () {
          const interceptor = yield* Effect.serviceOption(ToolInterceptorTag)
          const queue = yield* Queue.unbounded<QueueItem>()
          const stateRef = yield* Ref.make(opts?.initialState ?? initialReactorState())
          const turnId = opts?.turnId ?? `turn-${Date.now()}`

          // Replay context
          const priorToolCallIds = new Set(opts?.initialState?.toolCallMap.keys() ?? [])
          const priorOutcomes = opts?.initialState?.toolOutcomes ?? new Map()

          // Replay-aware ID generator
          // Fix 1: Pass this into createParser so IDs are consistent during replay
          const generateId = (() => {
            const priorIds = [...(opts?.initialState?.toolCallMap.keys() ?? [])]
            let ordinal = 0
            return () => {
              if (ordinal < priorIds.length) return priorIds[ordinal++]
              return `call-${++ordinal}-${Date.now().toString(36)}`
            }
          })()

          // Parser state — Fix 1: pass generateId so replay uses prior IDs
          const parser = createParser(generateId)

          // Accumulators for in-progress tool calls
          const activeInvokes = new Map<string, {
            tagName: string
            toolName: string
            group: string
            parameters: Map<string, ParsedParameter>
            hasFilter: boolean
            filterQuery?: string
          }>()

          // ---------------------------------------------------------------
          // Reactor glue
          // ---------------------------------------------------------------

          function emitAndFold(
            state: ReactorState,
            event: RuntimeEvent,
          ): Effect.Effect<ReactorState> {
            return Effect.gen(function* () {
              yield* Queue.offer(queue, event)
              return foldReactorState(state, event)
            })
          }

          const createMessageId = createShortId
          let activeProseMessageId: string | null = null

          // Process a single parser event
          function processEvent(
            state: ReactorState,
            event: ParserEvent,
          ): Effect.Effect<ReactorState> {
            return Effect.gen(function* () {
              let currentState = state

              // Handle structural events
              if (isStructuralEvent(event)) {
                currentState = yield* processStructuralEvent(currentState, event)
                return currentState
              }

              // Handle parse events
              switch (event._tag) {
                case 'InvokeStarted': {
                  if (hasPriorOutcome(priorOutcomes, event.toolCallId)) break
                  if (isInFlight(priorToolCallIds, priorOutcomes, event.toolCallId)) break

                  const registered = config.tools.get(event.toolTag)
                  if (!registered) break

                  activeInvokes.set(event.toolCallId, {
                    tagName: event.toolTag,
                    toolName: event.toolName,
                    group: event.group,
                    parameters: new Map(),
                    hasFilter: false,
                  })

                  currentState = yield* emitAndFold(currentState, {
                    _tag: 'ToolInputStarted',
                    toolCallId: event.toolCallId,
                    tagName: event.toolTag,
                    toolName: event.toolName,
                    group: event.group,
                  })

                  break
                }

                case 'ParameterStarted': {
                  const invoke = activeInvokes.get(event.toolCallId)
                  if (!invoke) break

                  invoke.parameters.set(event.parameterName, {
                    name: event.parameterName,
                    value: '',
                    isComplete: false,
                  })
                  break
                }

                case 'ParameterChunk': {
                  // Fix 3: Only accumulate; emit full value in ParameterComplete
                  const invoke = activeInvokes.get(event.toolCallId)
                  if (!invoke) break

                  const param = invoke.parameters.get(event.parameterName)
                  if (!param) break

                  param.value += event.text
                  break
                }

                case 'ParameterComplete': {
                  const invoke = activeInvokes.get(event.toolCallId)
                  if (!invoke) break

                  const param = invoke.parameters.get(event.parameterName)
                  if (!param) break

                  param.isComplete = true

                  // Fix 3: Emit complete field value now that we have the full accumulated value
                  currentState = yield* emitAndFold(currentState, {
                    _tag: 'ToolInputFieldValue',
                    toolCallId: event.toolCallId,
                    field: event.parameterName,
                    value: param.value,
                  })
                  break
                }

                case 'FilterStarted': {
                  const invoke = activeInvokes.get(event.toolCallId)
                  if (!invoke) break

                  invoke.hasFilter = true
                  invoke.filterQuery = ''
                  break
                }

                case 'FilterChunk': {
                  const invoke = activeInvokes.get(event.toolCallId)
                  if (!invoke || !invoke.hasFilter) break

                  invoke.filterQuery = (invoke.filterQuery ?? '') + event.text
                  break
                }

                case 'FilterComplete': {
                  const invoke = activeInvokes.get(event.toolCallId)
                  if (!invoke || !invoke.hasFilter) break

                  invoke.filterQuery = event.query
                  break
                }

                case 'InvokeComplete': {
                  const invoke = activeInvokes.get(event.toolCallId)
                  if (!invoke) break

                  activeInvokes.delete(event.toolCallId)

                  // Replay: outcome known → suppress
                  if (hasPriorOutcome(priorOutcomes, event.toolCallId)) {
                    cleanupToolCallState(event.toolCallId)
                    break
                  }

                  // Dead tool calls: skip dispatch
                  if (currentState.deadToolCalls.has(event.toolCallId)) {
                    cleanupToolCallState(event.toolCallId)
                    break
                  }

                  const registered = config.tools.get(invoke.tagName)
                  if (!registered) {
                    cleanupToolCallState(event.toolCallId)
                    break
                  }

                  // Build parsed invoke
                  const parsedInvoke: ParsedInvoke = {
                    tagName: invoke.tagName,
                    toolCallId: event.toolCallId,
                    parameters: invoke.parameters,
                    filter: invoke.filterQuery,
                  }

                  // Get derived parameter schema
                  const toolSchema = toolSchemas.get(invoke.tagName)

                  // Build input from parameters + schema
                  let input: Record<string, unknown>
                  try {
                    if (toolSchema) {
                      input = buildInput(parsedInvoke, toolSchema.parameters)
                    } else {
                      // No schema derived — should not happen, but handle gracefully
                      throw new Error(`No parameter schema for tool '${invoke.tagName}'`)
                    }
                  } catch (e) {
                    const msg = e instanceof Error ? e.message : String(e)
                    currentState = yield* emitAndFold(currentState, {
                      _tag: 'ToolInputParseError',
                      toolCallId: event.toolCallId,
                      tagName: invoke.tagName,
                      toolName: registered.tool.name,
                      group: registered.groupName,
                      error: {
                        _tag: 'ToolValidationFailed',
                        id: event.toolCallId,
                        tagName: invoke.tagName,
                        detail: `Input building failed: ${msg}`,
                      } satisfies ParseErrorDetail,
                    })
                    cleanupToolCallState(event.toolCallId)
                    break
                  }

                  // Build dispatch context
                  const dispatchCtx: DispatchContext = {
                    tools: config.tools,
                    interceptor: Option.getOrUndefined(interceptor),
                    toolContext: {
                      emit: (value: unknown) => Queue.offer(queue, {
                        _tag: 'ToolEmission',
                        toolCallId: event.toolCallId,
                        value,
                      } as RuntimeEvent),
                    },
                    emit: (ev) => Effect.gen(function* () {
                      currentState = yield* emitAndFold(currentState, ev)

                      // After successful execution, handle output query and persistence
                      if (ev._tag === 'ToolExecutionEnded' && ev.result._tag === 'Success') {
                        const output = ev.result.output
                        const query = invoke.filterQuery ?? null

                        // Persist result to file
                        const resultPath = persistResult(output, turnId, event.toolCallId)

                        // Query and render filtered result
                        const { filtered, isPartial } = queryOutput(output, query, resultPath)
                        const contentParts = renderFilteredResult(
                          registered.tool.name,
                          filtered,
                          isPartial,
                          resultPath
                        )

                        // Emit observation with filtered content
                        currentState = yield* emitAndFold(currentState, {
                          _tag: 'ToolObservation',
                          toolCallId: event.toolCallId,
                          tagName: invoke.tagName,
                          query,
                          content: contentParts,
                        })
                      }
                    }),
                  }

                  // Emit ToolInputReady
                  if (!isInFlight(priorToolCallIds, priorOutcomes, event.toolCallId)) {
                    currentState = yield* emitAndFold(currentState, {
                      _tag: 'ToolInputReady',
                      toolCallId: event.toolCallId,
                      input,
                    })
                  }

                  // Dispatch tool
                  const result: DispatchResult = yield* dispatchTool({
                    tagName: invoke.tagName,
                    toolCallId: event.toolCallId,
                    input,
                  }, dispatchCtx)

                  if (result._tag === 'ParseError') {
                    currentState = yield* emitAndFold(currentState, {
                      _tag: 'ToolInputParseError',
                      toolCallId: event.toolCallId,
                      tagName: invoke.tagName,
                      toolName: registered.tool.name,
                      group: registered.groupName,
                      error: result.error,
                    })
                  } else {
                    // Check for rejection
                    const outcome = currentState.toolOutcomes.get(event.toolCallId)
                    if (outcome?._tag === 'Completed' && outcome.result._tag === 'Rejected') {
                      currentState = yield* emitAndFold(currentState, {
                        _tag: 'TurnEnd',
                        result: { _tag: 'GateRejected', rejection: outcome.result.rejection },
                      })
                    }
                  }

                  cleanupToolCallState(event.toolCallId)
                  break
                }

                case 'ParseError': {
                  const error = event.error

                  if (error._tag === 'UnclosedThink') {
                    currentState = yield* emitAndFold(currentState, {
                      _tag: 'StructuralParseError',
                      error,
                    })
                    break
                  }

                  // Tool-level parse errors
                  if ('id' in error && 'tagName' in error) {
                    if (currentState.deadToolCalls.has(error.id)) break
                    if (hasPriorOutcome(priorOutcomes, error.id)) break

                    const registered = config.tools.get(error.tagName)
                    if (registered) {
                      currentState = yield* emitAndFold(currentState, {
                        _tag: 'ToolInputParseError',
                        toolCallId: error.id,
                        tagName: error.tagName,
                        toolName: registered.tool.name,
                        group: registered.groupName,
                        error,
                      })
                    }
                  }
                  break
                }
              }

              return currentState
            })
          }

          // Process structural events
          function processStructuralEvent(
            state: ReactorState,
            event: StructuralEvent,
          ): Effect.Effect<ReactorState> {
            return Effect.gen(function* () {
              let currentState = state

              switch (event._tag) {
                case 'ProseChunk': {
                  let id = activeProseMessageId
                  if (!id) {
                    id = createMessageId()
                    activeProseMessageId = id
                    currentState = yield* emitAndFold(currentState, {
                      _tag: 'MessageStart',
                      id,
                      to: null,
                    })
                  }
                  currentState = yield* emitAndFold(currentState, {
                    _tag: 'MessageChunk',
                    id,
                    text: event.text,
                  })
                  break
                }

                case 'ProseEnd': {
                  const id = activeProseMessageId
                  if (id) {
                    currentState = yield* emitAndFold(currentState, {
                      _tag: 'MessageEnd',
                      id,
                    })
                    activeProseMessageId = null
                  }
                  break
                }

                case 'LensStart': {
                  currentState = yield* emitAndFold(currentState, {
                    _tag: 'LensStart',
                    name: event.name,
                  })
                  break
                }

                case 'LensChunk': {
                  currentState = yield* emitAndFold(currentState, {
                    _tag: 'LensChunk',
                    text: event.text,
                  })
                  break
                }

                case 'LensEnd': {
                  currentState = yield* emitAndFold(currentState, {
                    _tag: 'LensEnd',
                    name: event.name,
                    content: event.content,
                  })
                  break
                }

                case 'MessageStart': {
                  currentState = yield* emitAndFold(currentState, {
                    _tag: 'MessageStart',
                    id: event.id,
                    to: event.to,
                  })
                  break
                }

                case 'MessageChunk': {
                  currentState = yield* emitAndFold(currentState, {
                    _tag: 'MessageChunk',
                    id: event.id,
                    text: event.text,
                  })
                  break
                }

                case 'MessageEnd': {
                  currentState = yield* emitAndFold(currentState, {
                    _tag: 'MessageEnd',
                    id: event.id,
                  })
                  break
                }

                case 'TurnControl': {
                  currentState = yield* emitAndFold(currentState, {
                    _tag: 'TurnEnd',
                    result: {
                      _tag: 'Success',
                      turnControl: { target: event.target },
                      termination: event.termination,
                    },
                  })
                  break
                }


              }

              return currentState
            })
          }

          function cleanupToolCallState(toolCallId: string): void {
            activeInvokes.delete(toolCallId)
          }

          // ---------------------------------------------------------------
          // Producer fiber: tokenize → parse → react → dispatch
          // ---------------------------------------------------------------

          const producer = Effect.gen(function* () {
            let tokenBuffer: Token[] = []

            // Create tokenizer ONCE — it must be persistent across stream chunks
            // so that tags split across chunks are correctly tokenized.
            const tokenizer = createTokenizer((token) => {
              tokenBuffer.push(token)
            })

            const flushTokensEffect = (): Effect.Effect<void> => Effect.gen(function* () {
              for (const token of tokenBuffer) {
                parser.pushToken(token)
              }
              tokenBuffer = []

              for (const event of parser.drain()) {
                let state = yield* Ref.get(stateRef)
                if (state.stopped) break
                state = yield* processEvent(state, event)
                yield* Ref.set(stateRef, state)
              }
            })

            yield* textStream.pipe(
              Stream.mapError((e) => new TurnEngineCrash(e.message, e)),
              Stream.runForEach((chunk) =>
                Effect.gen(function* () {
                  const state = yield* Ref.get(stateRef)

                  if (state.stopped) {
                    return yield* Effect.fail(new TurnEngineCrash('__runaway_abort__'))
                  }

                  tokenizer.push(chunk)
                  yield* flushTokensEffect()
                }),
              ),
            )

            // End tokenizer and flush parser
            tokenizer.end()
            let state = yield* Ref.get(stateRef)
            if (!state.stopped) {
              parser.end()
              for (const event of parser.drain()) {
                if (state.stopped) break
                state = yield* processEvent(state, event)
                yield* Ref.set(stateRef, state)
              }
            }

            if (!state.stopped) {
              yield* Queue.offer(queue, {
                _tag: 'TurnEnd',
                result: { _tag: 'Success', turnControl: null, termination: 'natural' },
              } satisfies RuntimeEvent)
            }

            yield* Queue.offer(queue, END)
          }).pipe(
            Effect.catchAll((crash) =>
              Effect.gen(function* () {
                if (crash instanceof TurnEngineCrash) {
                  if (crash.message === '__runaway_abort__') {
                    yield* Queue.offer(queue, END)
                    return
                  }
                  yield* Queue.offer(queue, {
                    _tag: 'TurnEnd',
                    result: { _tag: 'Failure', error: crash.message },
                  } satisfies RuntimeEvent)
                }
                yield* Queue.offer(queue, END)
              }),
            ),
            Effect.catchAllCause((cause) =>
              Effect.gen(function* () {
                const defect = Cause.squash(cause)
                const message = describeDefect(defect)
                yield* Queue.offer(queue, {
                  _tag: 'TurnEnd',
                  result: { _tag: 'Failure', error: `Tool defect: ${message}` },
                } satisfies RuntimeEvent)
                yield* Queue.offer(queue, END)
              }),
            ),
          )

          yield* Effect.forkScoped(producer)

          return Stream.fromQueue(queue).pipe(
            Stream.takeWhile((item): item is RuntimeEvent => item !== END),
          )
        }),
      )
    },
  }
}

// =============================================================================
// Type guards
// =============================================================================

function isStructuralEvent(event: ParserEvent): event is StructuralEvent {
  return event._tag === 'ProseChunk' ||
    event._tag === 'ProseEnd' ||
    event._tag === 'LensStart' ||
    event._tag === 'LensChunk' ||
    event._tag === 'LensEnd' ||
    event._tag === 'MessageStart' ||
    event._tag === 'MessageChunk' ||
    event._tag === 'MessageEnd' ||
    event._tag === 'TurnControl'
}

// =============================================================================
// Replay helpers
// =============================================================================

function hasPriorOutcome(priorOutcomes: ReadonlyMap<string, unknown>, toolCallId: string): boolean {
  return priorOutcomes.has(toolCallId)
}

function isInFlight(
  priorToolCallIds: ReadonlySet<string>,
  priorOutcomes: ReadonlyMap<string, unknown>,
  toolCallId: string,
): boolean {
  return priorToolCallIds.has(toolCallId) && !priorOutcomes.has(toolCallId)
}
