/**
 * TurnEngine — orchestration layer.
 *
 * Consumes a stream of ResponseStreamEvent values from the codec, translates
 * them into engine events, dispatches tools, and emits engine events to the
 * consumer queue.
 */

import { Effect, Stream, Ref, Option, Queue, Cause } from "effect"
import type { ResponseStreamEvent, ResponseUsage } from '@magnitudedev/codecs'
import type { ToolContext } from '@magnitudedev/tools'

import type { TurnEngineEvent, EngineState, RegisteredTool } from './types'
import { TurnEngineCrash, ToolInterceptorTag } from './types'
import { initialEngineState, foldEngineState } from './engine-state'
import { dispatchTool, type DispatchContext } from './dispatcher'

// =============================================================================
// Sentinel for end-of-stream
// =============================================================================

const END = Symbol('END')
type QueueItem = TurnEngineEvent | typeof END

function describeDefect(defect: unknown): string {
  if (defect instanceof Error) {
    const stack = defect.stack
    if (stack) return stack
    return defect.message
  }
  if (typeof defect === 'string') return defect
  if (typeof defect === 'object' && defect !== null) {
    const parts: string[] = []
    if ('_tag' in defect) parts.push(`[${(defect as Record<string, unknown>)._tag}]`)
    if ('name' in defect) parts.push(`${(defect as Record<string, unknown>).name}`)
    if ('message' in defect) parts.push(`${(defect as Record<string, unknown>).message}`)
    if (parts.length > 0) return parts.join(' ')
    try { return JSON.stringify(defect) } catch { /* ignore */ }
  }
  return String(defect)
}

// =============================================================================
// Public API
// =============================================================================

export interface TurnEngine {
  readonly streamWith: <E extends { message: string }>(
    eventStream: Stream.Stream<ResponseStreamEvent, E>,
    opts?: {
      readonly initialState?: EngineState
    },
  ) => Stream.Stream<TurnEngineEvent, TurnEngineCrash>
}

export interface TurnEngineConfig {
  readonly tools: ReadonlyMap<string, RegisteredTool>
  readonly messageDestination: string
  readonly thoughtKind?: string
}

/**
 * Create a TurnEngine from configuration.
 */
export function createTurnEngine(config: TurnEngineConfig): TurnEngine {
  function setAtPath(target: Record<string, unknown>, path: readonly string[], value: unknown): void {
    if (path.length === 0) return
    let cursor: Record<string, unknown> = target
    for (let i = 0; i < path.length - 1; i++) {
      const segment = path[i]
      const next = cursor[segment]
      if (next === undefined || next === null || typeof next !== 'object') {
        cursor[segment] = {}
      }
      cursor = cursor[segment] as Record<string, unknown>
    }
    cursor[path[path.length - 1]] = value
  }

  function makeCompletedOutcome(toolCallsCount: number) {
    return { _tag: 'Completed' as const, toolCallsCount }
  }

  return {
    streamWith<E extends { message: string }>(
      eventStream: Stream.Stream<ResponseStreamEvent, E>,
      opts?: { readonly initialState?: EngineState },
    ): Stream.Stream<TurnEngineEvent, TurnEngineCrash> {
      return Stream.unwrapScoped(
        Effect.gen(function* () {
          const interceptor = yield* Effect.serviceOption(ToolInterceptorTag)
          const queue = yield* Queue.unbounded<QueueItem>()
          const stateRef = yield* Ref.make(opts?.initialState ?? initialEngineState())
          const priorToolCallIds = new Set(opts?.initialState?.toolCallMap.keys() ?? [])
          const priorOutcomes: ReadonlyMap<string, unknown> =
            opts?.initialState?.toolOutcomes ?? new Map()

          const activeInvokes = new Map<string, { toolName: string; group: string }>()
          const assembledInputs = new Map<string, Record<string, unknown>>()
          let messageOrdinal = 0
          let activeMessageId: string | null = null
          let lastUsage: ResponseUsage | null = null

          function emitAndFold(
            state: EngineState,
            event: TurnEngineEvent,
          ): Effect.Effect<EngineState> {
            return Effect.gen(function* () {
              yield* Queue.offer(queue, event)
              return foldEngineState(state, event)
            })
          }

          function hasPriorOutcome(toolCallId: string): boolean {
            return priorOutcomes.has(toolCallId)
          }

          function isInFlight(toolCallId: string): boolean {
            return priorToolCallIds.has(toolCallId) && !priorOutcomes.has(toolCallId)
          }

          function currentToolCallsCount(state: EngineState): number {
            return state.toolCallMap.size
          }

          function processEngineEvent(
            state: EngineState,
            event: TurnEngineEvent,
          ): Effect.Effect<EngineState> {
            return Effect.gen(function* () {
              let currentState = state

              switch (event._tag) {
                case 'ToolInputStarted': {
                  if (hasPriorOutcome(event.toolCallId)) break
                  if (isInFlight(event.toolCallId)) {
                    activeInvokes.set(event.toolCallId, {
                      toolName: event.toolName,
                      group: event.group,
                    })
                    break
                  }
                  activeInvokes.set(event.toolCallId, {
                    toolName: event.toolName,
                    group: event.group,
                  })
                  currentState = yield* emitAndFold(currentState, event)
                  break
                }

                case 'ToolInputFieldChunk':
                case 'ToolInputFieldComplete': {
                  if (hasPriorOutcome(event.toolCallId)) break
                  if (isInFlight(event.toolCallId)) break
                  if (currentState.deadToolCalls.has(event.toolCallId)) break
                  currentState = yield* emitAndFold(currentState, event)
                  break
                }

                case 'ToolInputDecodeFailure': {
                  if (hasPriorOutcome(event.toolCallId)) break
                  activeInvokes.delete(event.toolCallId)
                  assembledInputs.delete(event.toolCallId)
                  currentState = yield* emitAndFold(currentState, event)
                  if (!currentState.stopped && lastUsage !== null) {
                    currentState = yield* emitAndFold(currentState, {
                      _tag: 'TurnEnd',
                      outcome: {
                        _tag: 'ToolInputDecodeFailure',
                        toolCallId: event.toolCallId,
                        toolName: event.toolName,
                        detail: event.detail,
                      },
                      usage: lastUsage,
                    })
                  }
                  break
                }

                case 'TurnStructureDecodeFailure': {
                  currentState = yield* emitAndFold(currentState, event)
                  if (!currentState.stopped && lastUsage !== null) {
                    currentState = yield* emitAndFold(currentState, {
                      _tag: 'TurnEnd',
                      outcome: {
                        _tag: 'TurnStructureDecodeFailure',
                        detail: event.detail,
                      },
                      usage: lastUsage,
                    })
                  }
                  break
                }

                case 'ToolInputReady': {
                  const invoke = activeInvokes.get(event.toolCallId)

                  if (hasPriorOutcome(event.toolCallId)) {
                    activeInvokes.delete(event.toolCallId)
                    assembledInputs.delete(event.toolCallId)
                    break
                  }

                  if (currentState.deadToolCalls.has(event.toolCallId)) {
                    activeInvokes.delete(event.toolCallId)
                    assembledInputs.delete(event.toolCallId)
                    break
                  }

                  if (!invoke) {
                    activeInvokes.delete(event.toolCallId)
                    assembledInputs.delete(event.toolCallId)
                    const detail = {
                      kind: 'EngineProtocolViolation',
                      message: `ToolInputReady for unknown toolCallId '${event.toolCallId}' (no preceding ToolInputStarted)`,
                    }
                    currentState = yield* emitAndFold(currentState, {
                      _tag: 'TurnStructureDecodeFailure',
                      detail,
                    })
                    if (!currentState.stopped && lastUsage !== null) {
                      currentState = yield* emitAndFold(currentState, {
                        _tag: 'TurnEnd',
                        outcome: { _tag: 'TurnStructureDecodeFailure', detail },
                        usage: lastUsage,
                      })
                    }
                    break
                  }

                  const inFlight = isInFlight(event.toolCallId)

                  if (!inFlight) {
                    currentState = yield* emitAndFold(currentState, event)
                  }

                  const dispatchCtx: DispatchContext = {
                    tools: config.tools,
                    interceptor: Option.getOrUndefined(interceptor),
                    toolContext: {
                      emit: (value: unknown) =>
                        Queue.offer(queue, {
                          _tag: 'ToolEmission',
                          toolCallId: event.toolCallId,
                          value,
                        } as TurnEngineEvent),
                    } satisfies ToolContext<unknown>,
                    emit: (ev: TurnEngineEvent) => Effect.gen(function* () {
                      currentState = yield* emitAndFold(currentState, ev)
                    }),
                  }

                  const result = yield* dispatchTool(
                    {
                      toolName: invoke.toolName,
                      toolCallId: event.toolCallId,
                      input: event.input,
                    },
                    dispatchCtx,
                  )

                  if (result._tag === 'DecodeFailure') {
                    currentState = yield* emitAndFold(currentState, {
                      _tag: 'ToolInputDecodeFailure',
                      toolCallId: event.toolCallId,
                      toolName: invoke.toolName,
                      group: invoke.group,
                      detail: result.detail,
                    })
                    if (!currentState.stopped && lastUsage !== null) {
                      currentState = yield* emitAndFold(currentState, {
                        _tag: 'TurnEnd',
                        outcome: {
                          _tag: 'ToolInputDecodeFailure',
                          toolCallId: event.toolCallId,
                          toolName: invoke.toolName,
                          detail: result.detail,
                        },
                        usage: lastUsage,
                      })
                    }
                  } else {
                    const outcome = currentState.toolOutcomes.get(event.toolCallId)
                    if (outcome?._tag === 'Completed' && outcome.result._tag === 'Rejected' && !currentState.stopped && lastUsage !== null) {
                      currentState = yield* emitAndFold(currentState, {
                        _tag: 'TurnEnd',
                        outcome: {
                          _tag: 'GateRejected',
                          toolCallId: event.toolCallId,
                          toolName: invoke.toolName,
                        },
                        usage: lastUsage,
                      })
                    }
                  }

                  activeInvokes.delete(event.toolCallId)
                  assembledInputs.delete(event.toolCallId)
                  break
                }

                case 'ThoughtStart':
                case 'ThoughtChunk':
                case 'ThoughtEnd':
                case 'MessageStart':
                case 'MessageChunk':
                case 'MessageEnd':
                case 'TurnEnd':
                case 'ToolExecutionStarted':
                case 'ToolExecutionEnded':
                case 'ToolEmission':
                  currentState = yield* emitAndFold(currentState, event)
                  break

                default: {
                  const _exhaustive: never = event
                  void _exhaustive
                  break
                }
              }

              return currentState
            })
          }

          function processResponseEvent(
            state: EngineState,
            event: ResponseStreamEvent,
          ): Effect.Effect<EngineState> {
            return Effect.gen(function* () {
              let currentState = state

              switch (event.type) {
                case 'thought_start':
                  return yield* processEngineEvent(currentState, {
                    _tag: 'ThoughtStart',
                    kind: config.thoughtKind ?? 'reasoning',
                  })
                case 'thought_delta':
                  return yield* processEngineEvent(currentState, {
                    _tag: 'ThoughtChunk',
                    text: event.text,
                  })
                case 'thought_end':
                  return yield* processEngineEvent(currentState, { _tag: 'ThoughtEnd' })

                case 'message_start': {
                  const id = `msg-${++messageOrdinal}-${Date.now().toString(36)}`
                  activeMessageId = id
                  return yield* processEngineEvent(currentState, {
                    _tag: 'MessageStart',
                    id,
                    to: config.messageDestination,
                  })
                }

                case 'message_delta':
                  if (activeMessageId === null) return currentState
                  return yield* processEngineEvent(currentState, {
                    _tag: 'MessageChunk',
                    id: activeMessageId,
                    text: event.text,
                  })

                case 'message_end':
                  if (activeMessageId === null) return currentState
                  currentState = yield* processEngineEvent(currentState, {
                    _tag: 'MessageEnd',
                    id: activeMessageId,
                  })
                  activeMessageId = null
                  return currentState

                case 'tool_call_start':
                  assembledInputs.set(event.toolCallId, {})
                  return yield* processEngineEvent(currentState, {
                    _tag: 'ToolInputStarted',
                    toolCallId: event.toolCallId,
                    toolName: event.toolName,
                    group: 'default',
                  })

                case 'tool_call_field_start':
                  return currentState

                case 'tool_call_field_delta':
                  return yield* processEngineEvent(currentState, {
                    _tag: 'ToolInputFieldChunk',
                    toolCallId: event.toolCallId,
                    field: (event.path[0] ?? '') as never,
                    path: event.path as never,
                    delta: event.delta,
                  })

                case 'tool_call_field_end': {
                  const call = assembledInputs.get(event.toolCallId)
                  if (call) {
                    setAtPath(call, event.path, event.value)
                  }
                  return yield* processEngineEvent(currentState, {
                    _tag: 'ToolInputFieldComplete',
                    toolCallId: event.toolCallId,
                    field: (event.path[0] ?? '') as never,
                    path: event.path as never,
                    value: event.value,
                  })
                }

                case 'tool_call_end': {
                  const input = assembledInputs.get(event.toolCallId)
                  if (!input) return currentState
                  return yield* processEngineEvent(currentState, {
                    _tag: 'ToolInputReady',
                    toolCallId: event.toolCallId,
                    input,
                  })
                }

                case 'response_done': {
                  lastUsage = event.usage
                  const outcome =
                    event.reason === 'stop' || event.reason === 'tool_calls'
                      ? makeCompletedOutcome(currentToolCallsCount(currentState))
                      : event.reason === 'length'
                        ? { _tag: 'OutputTruncated' as const }
                        : event.reason === 'content_filter'
                          ? { _tag: 'ContentFiltered' as const }
                          : { _tag: 'EngineDefect' as const, message: 'Unknown finish reason' }

                  return yield* processEngineEvent(currentState, {
                    _tag: 'TurnEnd',
                    outcome,
                    usage: event.usage,
                  })
                }
              }
            })
          }

          const producer = Effect.gen(function* () {
            yield* eventStream.pipe(
              Stream.mapError((e) => new TurnEngineCrash(e.message, e)),
              Stream.runForEach((event) =>
                Effect.gen(function* () {
                  const state = yield* Ref.get(stateRef)
                  if (state.stopped) {
                    return yield* Effect.fail(new TurnEngineCrash('__runaway_abort__'))
                  }
                  const next = yield* processResponseEvent(state, event)
                  yield* Ref.set(stateRef, next)
                }),
              ),
            )

            yield* Queue.offer(queue, END)
          }).pipe(
            Effect.catchAll((crash) =>
              Effect.gen(function* () {
                if (crash instanceof TurnEngineCrash) {
                  if (crash.message === '__runaway_abort__') {
                    yield* Queue.offer(queue, END)
                    return
                  }
                  if (lastUsage !== null) {
                    yield* Queue.offer(queue, {
                      _tag: 'TurnEnd',
                      outcome: { _tag: 'EngineDefect', message: crash.message },
                      usage: lastUsage,
                    } satisfies TurnEngineEvent)
                  }
                }
                yield* Queue.offer(queue, END)
              }),
            ),
            Effect.catchAllCause((cause) =>
              Effect.gen(function* () {
                const defect = Cause.squash(cause)
                const message = describeDefect(defect)
                if (lastUsage !== null) {
                  yield* Queue.offer(queue, {
                    _tag: 'TurnEnd',
                    outcome: { _tag: 'EngineDefect', message },
                    usage: lastUsage,
                  } satisfies TurnEngineEvent)
                }
                yield* Queue.offer(queue, END)
              }),
            ),
          )

          yield* Effect.forkScoped(producer)

          return Stream.fromQueue(queue).pipe(
            Stream.takeWhile((item): item is TurnEngineEvent => item !== END),
          )
        }),
      )
    },
  }
}
