/**
 * TurnEngine — orchestration layer.
 *
 * Wires text stream → tokenizer → parser → processEvent loop → event queue → consumer stream.
 * The parser emits TurnEngineEvent directly. The engine handles:
 *   - Replay guards (skip prior outcomes, re-run in-flight)
 *   - Forwarding events to the consumer queue
 *   - Dispatching tools when ToolInputReady arrives
 *   - Output observation (via dispatcher)
 *   - Error boundaries (crashes + defects)
 *
 * The engine does NO parsing. All parsing is done by the parser.
 */

import { Effect, Stream, Ref, Option, Queue, Cause } from "effect"
import type { ToolContext } from '@magnitudedev/tools'

import { createTokenizer } from '../tokenizer'
import { createParser } from '../parser/index'
import type { TurnEngineEvent, EngineState, RegisteredTool, ToolInterceptor } from '../types'
import { TurnEngineCrash, ToolInterceptorTag } from '../types'
import { initialEngineState, foldEngineState } from './engine-state'
import { dispatchTool, type DispatchContext } from './dispatcher'

// =============================================================================
// Sentinel for end-of-stream
// =============================================================================

const END = Symbol('END')
type QueueItem = TurnEngineEvent | typeof END

function describeDefect(defect: unknown): string {
  if (defect instanceof Error) {
    const msg = defect.message
    const stack = defect.stack
    if (stack) {
      return stack
    }
    return msg
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
    textStream: Stream.Stream<string, E>,
    opts?: {
      readonly initialState?: EngineState
      readonly turnId?: string
    },
  ) => Stream.Stream<TurnEngineEvent, TurnEngineCrash>
}

export interface TurnEngineConfig {
  readonly tools: ReadonlyMap<string, RegisteredTool>
  readonly defaultProseDest: string
  readonly resultsDir: string
}

/**
 * Create a TurnEngine from configuration.
 */
export function createTurnEngine(config: TurnEngineConfig): TurnEngine {
  return {
    streamWith<E extends { message: string }>(
      textStream: Stream.Stream<string, E>,
      opts?: { readonly initialState?: EngineState; readonly turnId?: string },
    ): Stream.Stream<TurnEngineEvent, TurnEngineCrash> {
      return Stream.unwrapScoped(
        Effect.gen(function* () {
          const interceptor = yield* Effect.serviceOption(ToolInterceptorTag)
          const queue = yield* Queue.unbounded<QueueItem>()
          const stateRef = yield* Ref.make(opts?.initialState ?? initialEngineState())
          const turnId = opts?.turnId ?? `turn-${Date.now()}`

          // Replay context
          const priorToolCallIds = new Set(opts?.initialState?.toolCallMap.keys() ?? [])
          const priorOutcomes: ReadonlyMap<string, unknown> =
            opts?.initialState?.toolOutcomes ?? new Map()

          // Replay-aware ID generator — yields prior IDs in order, then fresh ones
          const generateId = (() => {
            const priorIds = [...(opts?.initialState?.toolCallMap.keys() ?? [])]
            let ordinal = 0
            return () => {
              if (ordinal < priorIds.length) return priorIds[ordinal++]
              return `call-${++ordinal}-${Date.now().toString(36)}`
            }
          })()

          // Per-invoke tracking: tagName needed at ToolInputReady for dispatch
          // filterQuery: currently not surfaced by parser in events (tracked as null)
          const activeInvokes = new Map<string, { tagName: string; filterQuery: string | null; openSpan?: import('../types').SourceSpan }>()

          // Prose-to-message conversion state
          let proseMessageId: string | null = null
          let proseMessageOrdinal = 0

          // ---------------------------------------------------------------
          // Emit + fold helpers
          // ---------------------------------------------------------------

          function emitAndFold(
            state: EngineState,
            event: TurnEngineEvent,
          ): Effect.Effect<EngineState> {
            return Effect.gen(function* () {
              yield* Queue.offer(queue, event)
              return foldEngineState(state, event)
            })
          }

          // ---------------------------------------------------------------
          // Replay guards
          // ---------------------------------------------------------------

          function hasPriorOutcome(toolCallId: string): boolean {
            return priorOutcomes.has(toolCallId)
          }

          function isInFlight(toolCallId: string): boolean {
            return priorToolCallIds.has(toolCallId) && !priorOutcomes.has(toolCallId)
          }

          // ---------------------------------------------------------------
          // processEvent — single switch over TurnEngineEvent
          // ---------------------------------------------------------------

          function processEvent(
            state: EngineState,
            event: TurnEngineEvent,
          ): Effect.Effect<EngineState> {
            return Effect.gen(function* () {
              let currentState = state

              switch (event._tag) {
                // ----------------------------------------------------------
                // Tool input lifecycle — replay guards apply
                // ----------------------------------------------------------

                case 'ToolInputStarted': {
                  if (hasPriorOutcome(event.toolCallId)) break
                  if (isInFlight(event.toolCallId)) {
                    // In-flight: track but suppress the event
                    activeInvokes.set(event.toolCallId, { tagName: event.tagName, filterQuery: null, openSpan: event.openSpan })
                    break
                  }
                  activeInvokes.set(event.toolCallId, { tagName: event.tagName, filterQuery: null, openSpan: event.openSpan })
                  currentState = yield* emitAndFold(currentState, event)
                  break
                }

                case 'ToolInputFieldChunk': {
                  if (hasPriorOutcome(event.toolCallId)) break
                  if (isInFlight(event.toolCallId)) break
                  if (currentState.deadToolCalls.has(event.toolCallId)) break
                  currentState = yield* emitAndFold(currentState, event)
                  break
                }

                case 'ToolInputFieldComplete': {
                  if (hasPriorOutcome(event.toolCallId)) break
                  if (isInFlight(event.toolCallId)) break
                  if (currentState.deadToolCalls.has(event.toolCallId)) break
                  currentState = yield* emitAndFold(currentState, event)
                  break
                }

                case 'ToolParseError': {
                  if (hasPriorOutcome(event.toolCallId)) break
                  activeInvokes.delete(event.toolCallId)
                  currentState = yield* emitAndFold(currentState, event)
                  currentState = yield* emitAndFold(currentState, {
                    _tag: 'TurnEnd',
                    outcome: { _tag: 'ToolParseError', error: event },
                  })
                  break
                }

                case 'StructuralParseError': {
                  currentState = yield* emitAndFold(currentState, event)
                  currentState = yield* emitAndFold(currentState, {
                    _tag: 'TurnEnd',
                    outcome: { _tag: 'StructuralParseError', error: event },
                  })
                  break
                }

                case 'ToolInputReady': {
                  const invoke = activeInvokes.get(event.toolCallId)

                  if (hasPriorOutcome(event.toolCallId)) {
                    activeInvokes.delete(event.toolCallId)
                    break
                  }

                  if (currentState.deadToolCalls.has(event.toolCallId)) {
                    activeInvokes.delete(event.toolCallId)
                    break
                  }

                  if (!invoke) {
                    activeInvokes.delete(event.toolCallId)
                    const error = {
                      _tag: 'UnexpectedContent' as const,
                      context: 'engine',
                      detail: `ToolInputReady for unknown toolCallId '${event.toolCallId}'`,
                    }
                    const structuralParseErrorEvent = {
                      _tag: 'StructuralParseError' as const,
                      error,
                    }
                    currentState = yield* emitAndFold(currentState, structuralParseErrorEvent)
                    currentState = yield* emitAndFold(currentState, {
                      _tag: 'TurnEnd',
                      outcome: { _tag: 'StructuralParseError', error: structuralParseErrorEvent },
                    })
                    break
                  }

                  const inFlight = isInFlight(event.toolCallId)

                  // Emit ToolInputReady only for new (non-replay) calls
                  if (!inFlight) {
                    currentState = yield* emitAndFold(currentState, event)
                  }

                  // Dispatch tool
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
                      tagName: invoke.tagName,
                      toolCallId: event.toolCallId,
                      input: event.input,
                      filterQuery: invoke.filterQuery,
                      turnId,
                      resultsDir: config.resultsDir,
                    },
                    dispatchCtx,
                  )

                  if (result._tag === 'ParseError') {
                    const registered = config.tools.get(invoke.tagName)
                    const errorWithSpan = { ...result.error, primarySpan: invoke.openSpan } as ToolParseError
                    currentState = yield* emitAndFold(currentState, {
                      _tag: 'ToolParseError',
                      toolCallId: event.toolCallId,
                      tagName: invoke.tagName,
                      toolName: registered?.tool.name ?? invoke.tagName,
                      group: registered?.groupName ?? 'default',
                      error: errorWithSpan,
                    })
                  } else {
                    // Check for terminal dispatch outcomes → TurnEnd
                    const outcome = currentState.toolOutcomes.get(event.toolCallId)
                    if (outcome?._tag === 'Completed' && outcome.result._tag === 'Rejected') {
                      currentState = yield* emitAndFold(currentState, {
                        _tag: 'TurnEnd',
                        outcome: { _tag: 'GateRejected', rejection: outcome.result.rejection },
                      })
                    } else if (outcome?._tag === 'Completed' && outcome.result._tag === 'Error') {
                      currentState = yield* emitAndFold(currentState, {
                        _tag: 'TurnEnd',
                        outcome: { _tag: 'ToolExecutionError' },
                      })
                    }
                  }

                  activeInvokes.delete(event.toolCallId)
                  break
                }

                // ----------------------------------------------------------
                // Structural events — forward directly (no replay guards)
                // ----------------------------------------------------------

                case 'LensStart':
                case 'LensChunk':
                case 'LensEnd':
                case 'MessageStart':
                case 'MessageChunk':
                case 'MessageEnd':
                  currentState = yield* emitAndFold(currentState, event)
                  break

                case 'ProseChunk': {
                  if (config.defaultProseDest) {
                    if (!proseMessageId) {
                      proseMessageId = `prose-msg-${++proseMessageOrdinal}-${Date.now().toString(36)}`
                      currentState = yield* emitAndFold(currentState, {
                        _tag: 'MessageStart',
                        id: proseMessageId,
                        to: config.defaultProseDest,
                      })
                    }
                    currentState = yield* emitAndFold(currentState, {
                      _tag: 'MessageChunk',
                      id: proseMessageId,
                      text: event.text,
                    })
                  } else {
                    currentState = yield* emitAndFold(currentState, event)
                  }
                  break
                }

                case 'ProseEnd': {
                  if (config.defaultProseDest && proseMessageId) {
                    currentState = yield* emitAndFold(currentState, {
                      _tag: 'MessageEnd',
                      id: proseMessageId,
                    })
                    proseMessageId = null
                  } else {
                    currentState = yield* emitAndFold(currentState, event)
                  }
                  break
                }

                case 'TurnEnd':
                  currentState = yield* emitAndFold(currentState, event)
                  break

                // ----------------------------------------------------------
                // Execution events — emitted by dispatcher via emit callback,
                // should not arrive from the parser directly
                // ----------------------------------------------------------

                case 'ToolExecutionStarted':
                case 'ToolExecutionEnded':
                case 'ToolEmission':
                case 'ToolObservation':
                  // Forward if they somehow arrive (defensive)
                  currentState = yield* emitAndFold(currentState, event)
                  break

                default:
                  break
              }

              return currentState
            })
          }

          // ---------------------------------------------------------------
          // Producer fiber: tokenize → parse → processEvent → queue
          // ---------------------------------------------------------------

          const producer = Effect.gen(function* () {
            const parser = createParser(
              {
                tools: config.tools,
                generateId,
              },
              // onFilterReady callback — store filter query for dispatch
              (filterEvent) => {
                const invoke = activeInvokes.get(filterEvent.toolCallId)
                if (invoke) {
                  activeInvokes.set(filterEvent.toolCallId, { ...invoke, filterQuery: filterEvent.query })
                }
              },
            )

            const tokenizer = createTokenizer(
              (token) => { parser.pushToken(token) },
              new Set(config.tools.keys()),
              { toolKeyword: 'magnitude:invoke' },
            )

            yield* textStream.pipe(
              Stream.mapError((e) => new TurnEngineCrash(e.message, e)),
              Stream.runForEach((chunk) =>
                Effect.gen(function* () {
                  const state = yield* Ref.get(stateRef)
                  if (state.stopped) {
                    return yield* Effect.fail(new TurnEngineCrash('__runaway_abort__'))
                  }

                  tokenizer.push(chunk)

                  for (const event of parser.drain()) {
                    let s = yield* Ref.get(stateRef)
                    if (s.stopped) break
                    s = yield* processEvent(s, event)
                    yield* Ref.set(stateRef, s)
                  }
                }),
              ),
            )

            // End tokenizer + flush parser
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

            // Natural TurnEnd if not already stopped
            const finalState = yield* Ref.get(stateRef)
            if (!finalState.stopped) {
              yield* Queue.offer(queue, {
                _tag: 'TurnEnd',
                outcome: { _tag: 'Completed', turnControl: null, termination: 'natural' },
              } satisfies TurnEngineEvent)
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
                    outcome: { _tag: 'EngineDefect', message: crash.message, cause: crash.cause },
                  } satisfies TurnEngineEvent)
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
                  outcome: {
                    _tag: 'EngineDefect',
                    message,
                  },
                } satisfies TurnEngineEvent)
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
