/**
 * XmlRuntime — queue-based runtime that wires parser → reactor → stream.
 *
 * The reactor is internal. The only public API is createXmlRuntime → XmlRuntime.
 * Events are delivered via an Effect Queue for real-time streaming (no batching).
 *
 * For replay, consumers provide initialState with pre-populated toolOutcomes.
 * The reactor skips completed tools entirely and re-runs only the in-flight tool.
 */

import { Effect, Stream, Ref, Option, Queue, Scope, Cause } from "effect"

import { createStreamingXmlParser, defaultIdGenerator } from '../parser/streaming-xml-parser'
import { createShortId } from '../util'
import type { IdGenerator } from '../parser/streaming-xml-parser'
import type { ParseEvent } from '../parser/types'
import type {
  XmlRuntimeConfig,
  XmlRuntimeEvent,
  ToolInterceptor,
  ToolCallContext,
  RegisteredTool,
  ReactorState,
} from '../types'
import { XmlRuntimeCrash, ToolInterceptorTag } from '../types'
import { dispatchTool, type DispatchContext, type DispatchResult } from './tool-dispatcher'
import { buildInput } from './input-builder'
import { observeOutput } from '../output-query'
import { validateBinding, type TagSchema } from './binding-validator'
import { initialReactorState, foldReactorState } from './reactor-state'

// =============================================================================
// Sentinel for end-of-stream
// =============================================================================

const END = Symbol('END')
type QueueItem = XmlRuntimeEvent | typeof END

// =============================================================================
// Public API
// =============================================================================

export interface XmlRuntime {
  readonly streamWith: <E extends { message: string } = Error>(
    xmlStream: Stream.Stream<string, E>,
    opts?: { readonly initialState?: ReactorState },
  ) => Stream.Stream<XmlRuntimeEvent, XmlRuntimeCrash>
}

/**
 * Create an XML runtime from configuration.
 * Validates all bindings eagerly — throws on binding errors.
 */
export function createXmlRuntime(config: XmlRuntimeConfig): XmlRuntime {
  const tagSchemas = new Map<string, TagSchema>()
  for (const [tagName, reg] of config.tools) {
    const schema = validateBinding(tagName, reg.binding, reg.tool.inputSchema.ast)
    tagSchemas.set(tagName, schema)
  }

  return {
    streamWith<E extends { message: string } = Error>(
      xmlStream: Stream.Stream<string, E>,
      opts?: { readonly initialState?: ReactorState },
    ): Stream.Stream<XmlRuntimeEvent, XmlRuntimeCrash> {
      return Stream.unwrapScoped(
        Effect.gen(function* () {
          const interceptor = yield* Effect.serviceOption(ToolInterceptorTag)
          const queue = yield* Queue.unbounded<QueueItem>()
          const stateRef = yield* Ref.make(opts?.initialState ?? initialReactorState())

          // Replay context: which tool calls existed before this run
          const priorToolCallIds = new Set(opts?.initialState?.toolCallMap.keys() ?? [])
          const priorOutcomes = opts?.initialState?.toolOutcomes ?? new Map()

          // Build replay-aware ID generator.
          // On replay, the parser must produce the same toolCallIds as the first run
          // so the reactor can match them against prior outcomes. The replay state's
          // toolCallMap preserves IDs in insertion order — we yield those first,
          // then fall back to fresh cuids for any new tool calls beyond the replay point.
          const generateId: IdGenerator = (() => {
            const priorIds = [...(opts?.initialState?.toolCallMap.keys() ?? [])]
            let ordinal = 0
            return () => {
              if (ordinal < priorIds.length) return priorIds[ordinal++]
              return defaultIdGenerator()
            }
          })()

          // Build child tag map for parser
          const knownTags = new Set(config.tools.keys())
          const childTagMap = new Map<string, Set<string>>()
          for (const [tag, reg] of config.tools) {
            const valid = new Set<string>()
            if (reg.binding.childTags) reg.binding.childTags.forEach(ct => valid.add(ct.tag))
            if (reg.binding.children) reg.binding.children.forEach(c => valid.add(c.tag ?? c.field))
            if (reg.binding.childRecord) valid.add(reg.binding.childRecord.tag)
            childTagMap.set(tag, valid)
          }

          const parser = createStreamingXmlParser(
            knownTags,
            childTagMap,
            tagSchemas,
            generateId,
            config.defaultProseDest ?? 'user',
          )

          // ---------------------------------------------------------------
          // Text coalescing queue
          //
          // The reactor processes chars one at a time (for ref store correctness)
          // which produces per-character BodyChunk/ProseChunk events. The
          // coalescing queue accumulates consecutive text events and flushes
          // them as single chunk-sized events at natural boundaries.
          // ---------------------------------------------------------------

          type TextBuffer =
            | { _tag: 'body'; toolCallId: string; path: readonly string[]; field: string; text: string }
            | { _tag: 'prose'; patternId: string; text: string }
            | { _tag: 'lens'; text: string }
            | { _tag: 'message'; id: string; text: string }
          let textBuffer: TextBuffer | null = null

          function flushTextBuffer(): Effect.Effect<void> {
            if (!textBuffer) return Effect.void
            const buf = textBuffer
            textBuffer = null
            switch (buf._tag) {
              case 'body':
                return Queue.offer(queue, {
                  _tag: 'ToolInputBodyChunk',
                  toolCallId: buf.toolCallId,
                  path: buf.path,
                  field: buf.field,
                  text: buf.text,
                } satisfies XmlRuntimeEvent)
              case 'prose':
                return Queue.offer(queue, {
                  _tag: 'ProseChunk',
                  patternId: buf.patternId,
                  text: buf.text,
                } satisfies XmlRuntimeEvent)
              case 'lens':
                return Queue.offer(queue, {
                  _tag: 'LensChunk',
                  text: buf.text,
                } satisfies XmlRuntimeEvent)
              case 'message':
                return Queue.offer(queue, {
                  _tag: 'MessageChunk',
                  id: buf.id,
                  text: buf.text,
                } satisfies XmlRuntimeEvent)
            }
          }

          function offerCoalesced(event: XmlRuntimeEvent): Effect.Effect<void> {
            if (event._tag === 'ToolInputBodyChunk') {
              if (textBuffer && textBuffer._tag === 'body'
                && textBuffer.toolCallId === event.toolCallId) {
                textBuffer.text += event.text
                return Effect.void
              }
              return Effect.gen(function* () {
                yield* flushTextBuffer()
                textBuffer = { _tag: 'body', toolCallId: event.toolCallId, path: event.path, field: event.field, text: event.text }
              })
            }

            if (event._tag === 'ProseChunk') {
              if (textBuffer && textBuffer._tag === 'prose'
                && textBuffer.patternId === event.patternId) {
                textBuffer.text += event.text
                return Effect.void
              }
              return Effect.gen(function* () {
                yield* flushTextBuffer()
                textBuffer = { _tag: 'prose', patternId: event.patternId, text: event.text }
              })
            }

            if (event._tag === 'LensChunk') {
              if (textBuffer && textBuffer._tag === 'lens') {
                textBuffer.text += event.text
                return Effect.void
              }
              return Effect.gen(function* () {
                yield* flushTextBuffer()
                textBuffer = { _tag: 'lens', text: event.text }
              })
            }

            if (event._tag === 'MessageChunk') {
              if (textBuffer && textBuffer._tag === 'message'
                && textBuffer.id === event.id) {
                textBuffer.text += event.text
                return Effect.void
              }
              return Effect.gen(function* () {
                yield* flushTextBuffer()
                textBuffer = { _tag: 'message', id: event.id, text: event.text }
              })
            }

            return Effect.gen(function* () {
              yield* flushTextBuffer()
              yield* Queue.offer(queue, event)
            })
          }

          // ---------------------------------------------------------------
          // Reactor glue
          // ---------------------------------------------------------------

          function emitAndFold(
            state: ReactorState,
            event: XmlRuntimeEvent,
          ): Effect.Effect<ReactorState> {
            return Effect.gen(function* () {
              yield* offerCoalesced(event)
              return foldReactorState(state, event)
            })
          }

          const createMessageId = createShortId
          let activeProseMessageId: string | null = null
          function react(
            state: ReactorState,
            parseEvent: ParseEvent,
          ): Effect.Effect<ReactorState> {
            return reactImpl(
              state, parseEvent, emitAndFold,
              config, tagSchemas, Option.getOrUndefined(interceptor),
              priorToolCallIds, priorOutcomes,
              {
                get: () => activeProseMessageId,
                set: (id) => { activeProseMessageId = id },
                create: createMessageId,
              },
            )
          }

          // Producer fiber: parse → react → dispatch, offer events to queue.
          // Processes char by char so the reactor runs between each character,
          // ensuring the ref store is populated before the parser encounters <ref>.
          // Text events (BodyChunk, ProseChunk) are coalesced within each LLM chunk
          // so consumers see chunk-sized text instead of single characters.
          const producer = Effect.gen(function* () {
            yield* xmlStream.pipe(
              Stream.mapError((e) => new XmlRuntimeCrash(e.message, e)),
              Stream.runForEachWhile((chunk) =>
                Effect.gen(function* () {
                  let state = yield* Ref.get(stateRef)
                  if (state.stopped) return false

                  for (const ch of chunk) {
                    if (state.stopped) break
                    const parseEvents = parser.processChunk(ch)
                    for (const pe of parseEvents) {
                      if (state.stopped) break
                      state = yield* react(state, pe)
                    }
                  }
                  // Flush any coalesced text at chunk boundary
                  yield* flushTextBuffer()
                  yield* Ref.set(stateRef, state)
                  return !state.stopped
                }),
              ),
            )

            // Flush parser
            let state = yield* Ref.get(stateRef)
            if (!state.stopped) {
              const flushEvents = parser.flush()
              for (const pe of flushEvents) {
                if (state.stopped) break
                state = yield* react(state, pe)
              }
              yield* flushTextBuffer()
              yield* Ref.set(stateRef, state)
            }

            if (!state.stopped) {
              yield* Queue.offer(queue, {
                _tag: 'TurnEnd',
                result: { _tag: 'Success', turnControl: null },
              } satisfies XmlRuntimeEvent)
            }

            yield* Queue.offer(queue, END)
          }).pipe(
            // Catch typed errors (XmlRuntimeCrash)
            Effect.catchAll((crash) =>
              Effect.gen(function* () {
                if (crash instanceof XmlRuntimeCrash) {
                  yield* Queue.offer(queue, {
                    _tag: 'TurnEnd',
                    result: { _tag: 'Failure', error: crash.message },
                  } satisfies XmlRuntimeEvent)
                }
                yield* Queue.offer(queue, END)
              }),
            ),
            // Catch defects (e.g. Effect.promise rejections from buggy tools).
            // Without this, defects kill the fiber silently and the stream hangs forever.
            Effect.catchAllCause((cause) =>
              Effect.gen(function* () {
                const defect = Cause.squash(cause)
                const message = defect instanceof Error ? defect.message : String(defect)
                yield* Queue.offer(queue, {
                  _tag: 'TurnEnd',
                  result: { _tag: 'Failure', error: `Tool defect (bug — use Effect.tryPromise, not Effect.promise): ${message}` },
                } satisfies XmlRuntimeEvent)
                yield* Queue.offer(queue, END)
              }),
            ),
          )

          // Fork producer — it runs in background, pushes events to queue
          yield* Effect.forkScoped(producer)

          // Return stream that reads from queue until END sentinel
          return Stream.fromQueue(queue).pipe(
            Stream.takeWhile((item): item is XmlRuntimeEvent => item !== END),
          )
        }),
      )
    },
  }
}

// =============================================================================
// Reactor implementation (internal)
// =============================================================================

/**
 * Check if a tool call should be fully suppressed (outcome known from prior run).
 */
function hasPriorOutcome(priorOutcomes: ReadonlyMap<string, unknown>, toolCallId: string): boolean {
  return priorOutcomes.has(toolCallId)
}

/**
 * Check if a tool call was started in a prior run but has no outcome —
 * this is the in-flight tool on replay.
 */
function isInFlight(
  priorToolCallIds: ReadonlySet<string>,
  priorOutcomes: ReadonlyMap<string, unknown>,
  toolCallId: string,
): boolean {
  return priorToolCallIds.has(toolCallId) && !priorOutcomes.has(toolCallId)
}

function reactImpl(
  state: ReactorState,
  parseEvent: ParseEvent,
  emitAndFold: (state: ReactorState, event: XmlRuntimeEvent) => Effect.Effect<ReactorState>,
  config: XmlRuntimeConfig,
  tagSchemas: ReadonlyMap<string, TagSchema>,
  interceptor: ToolInterceptor | undefined,
  priorToolCallIds: ReadonlySet<string>,
  priorOutcomes: ReadonlyMap<string, unknown>,
  proseMessage: {
    readonly get: () => string | null
    readonly set: (id: string | null) => void
    readonly create: () => string
  },
): Effect.Effect<ReactorState> {
  return Effect.gen(function* () {
    let currentState = state

    switch (parseEvent._tag) {
      case 'TagOpened': {
        const registered = config.tools.get(parseEvent.tagName)
        if (!registered) break

        // Replay: outcome known → suppress entirely
        if (hasPriorOutcome(priorOutcomes, parseEvent.toolCallId)) break
        // Replay: in-flight tool from prior run → input already emitted, suppress
        if (isInFlight(priorToolCallIds, priorOutcomes, parseEvent.toolCallId)) break

        // Emit ToolInputStarted
        currentState = yield* emitAndFold(currentState, {
          _tag: 'ToolInputStarted',
          toolCallId: parseEvent.toolCallId,
          tagName: parseEvent.tagName,
          toolName: registered.tool.name,
          group: registered.groupName,
        })

        // Emit ToolInputFieldValue for each bound attribute
        if (registered.binding.attributes) {
          for (const attrName of registered.binding.attributes) {
            const value = parseEvent.attributes.get(attrName)
            if (value !== undefined) {
              currentState = yield* emitAndFold(currentState, {
                _tag: 'ToolInputFieldValue',
                toolCallId: parseEvent.toolCallId,
                field: attrName,
                value,
              })
            }
          }
        }
        break
      }

      case 'BodyChunk': {
        if (currentState.deadToolCalls.has(parseEvent.toolCallId)) break
        if (hasPriorOutcome(priorOutcomes, parseEvent.toolCallId)) break
        if (isInFlight(priorToolCallIds, priorOutcomes, parseEvent.toolCallId)) break

        const tagNameForBody = currentState.toolCallMap.get(parseEvent.toolCallId)
        if (!tagNameForBody) break

        const tagSchema = tagSchemas.get(tagNameForBody)
        const bodyField = resolveBodyField(parseEvent.toolCallId, currentState, config.tools)

        if (bodyField) {
          currentState = yield* emitAndFold(currentState, {
            _tag: 'ToolInputBodyChunk',
            toolCallId: parseEvent.toolCallId,
            path: [bodyField],
            field: bodyField,
            text: parseEvent.text,
          })
        } else if (tagSchema && !tagSchema.acceptsBody) {
          const registered = config.tools.get(tagNameForBody)
          if (registered) {
            const call = makeCallContext(parseEvent.toolCallId, tagNameForBody, registered)
            currentState = yield* emitAndFold(currentState, {
              _tag: 'ToolInputParseError',
              toolCallId: parseEvent.toolCallId,
              tagName: tagNameForBody,
              toolName: registered.tool.name,
              group: registered.groupName,
              error: {
                _tag: 'UnexpectedBody',
                detail: `Tool <${tagNameForBody}> does not accept body content`,
                call,
              },
            })
          }
        }
        break
      }

      case 'ChildOpened': {
        if (currentState.deadToolCalls.has(parseEvent.parentToolCallId)) break
        if (hasPriorOutcome(priorOutcomes, parseEvent.parentToolCallId)) break
        if (isInFlight(priorToolCallIds, priorOutcomes, parseEvent.parentToolCallId)) break

        const childInfo = resolveChildBinding(parseEvent.parentToolCallId, parseEvent.childTagName, currentState, config.tools)
        if (childInfo) {
          const attrRecord: Record<string, string | number | boolean> = {}
          for (const [k, v] of parseEvent.attributes) attrRecord[k] = v

          currentState = yield* emitAndFold(currentState, {
            _tag: 'ToolInputChildStarted',
            toolCallId: parseEvent.parentToolCallId,
            field: childInfo.field,
            index: parseEvent.childIndex,
            attributes: attrRecord,
          })
        }
        break
      }

      case 'ChildBodyChunk': {
        if (currentState.deadToolCalls.has(parseEvent.parentToolCallId)) break
        if (hasPriorOutcome(priorOutcomes, parseEvent.parentToolCallId)) break
        if (isInFlight(priorToolCallIds, priorOutcomes, parseEvent.parentToolCallId)) break

        const childInfo = resolveChildBinding(parseEvent.parentToolCallId, parseEvent.childTagName, currentState, config.tools)
        if (childInfo?.bodyField) {
          currentState = yield* emitAndFold(currentState, {
            _tag: 'ToolInputBodyChunk',
            toolCallId: parseEvent.parentToolCallId,
            path: [childInfo.field, String(parseEvent.childIndex), childInfo.bodyField],
            field: childInfo.bodyField,
            text: parseEvent.text,
          })
        }
        break
      }

      case 'ChildComplete': {
        if (currentState.deadToolCalls.has(parseEvent.parentToolCallId)) break
        if (hasPriorOutcome(priorOutcomes, parseEvent.parentToolCallId)) break
        if (isInFlight(priorToolCallIds, priorOutcomes, parseEvent.parentToolCallId)) break

        const childInfo = resolveChildBinding(parseEvent.parentToolCallId, parseEvent.childTagName, currentState, config.tools)
        if (childInfo) {
          const value: Record<string, unknown> = {}
          if (childInfo.attributes) {
            for (const attrName of childInfo.attributes) {
              const v = parseEvent.attributes.get(attrName)
              if (v !== undefined) value[attrName] = v
            }
          }
          if (childInfo.bodyField) {
            value[childInfo.bodyField] = parseEvent.body.trim()
          }

          currentState = yield* emitAndFold(currentState, {
            _tag: 'ToolInputChildComplete',
            toolCallId: parseEvent.parentToolCallId,
            field: childInfo.field,
            index: parseEvent.childIndex,
            value,
          })
        }
        break
      }

      case 'TagClosed': {
        // Replay: outcome known → suppress entirely
        if (hasPriorOutcome(priorOutcomes, parseEvent.toolCallId)) break
        // Dead tool calls: skip dispatch
        if (currentState.deadToolCalls.has(parseEvent.toolCallId)) break

        const registered = config.tools.get(parseEvent.tagName)
        if (!registered) break

        // The dispatcher emits events via this callback — state stays in sync
        // automatically because every event goes through emitAndFold.
        const dispatchCtx: DispatchContext = {
          tools: config.tools,
          interceptor,
          emit: (event) => Effect.gen(function* () {
            currentState = yield* emitAndFold(currentState, event)
            if (event._tag === 'ToolExecutionEnded' && event.result._tag === 'Success') {
              currentState = yield* emitAndFold(currentState, {
                _tag: 'ToolObservation',
                toolCallId: event.toolCallId,
                tagName: event.result.outputTree.tag,
                query: event.result.query,
                content: observeOutput(event.result.outputTree.tree, event.result.query),
              })
            }
          }),
        }

        // Emit ToolInputReady (only if not in-flight replay — in-flight skips input events
        // but still dispatches)
        if (!isInFlight(priorToolCallIds, priorOutcomes, parseEvent.toolCallId)) {
          const rawInput = buildInput(parseEvent.element, registered.binding)
          currentState = yield* emitAndFold(currentState, {
            _tag: 'ToolInputReady',
            toolCallId: parseEvent.toolCallId,
            input: rawInput,
          })
        }

        // Dispatch tool — events emitted via callback, state folded automatically
        const result: DispatchResult = yield* dispatchTool(parseEvent.element, dispatchCtx)

        if (result._tag === 'ParseError') {
          const call = makeCallContext(parseEvent.toolCallId, parseEvent.tagName, registered)
          currentState = yield* emitAndFold(currentState, {
            _tag: 'ToolInputParseError',
            toolCallId: parseEvent.toolCallId,
            tagName: parseEvent.tagName,
            toolName: registered.tool.name,
            group: registered.groupName,
            error: { ...result.error, call },
          })
        } else {
          // Check for rejection (state already folded via emit callback)
          const outcome = currentState.toolOutcomes.get(parseEvent.toolCallId)
          if (outcome?._tag === 'Completed' && outcome.result._tag === 'Rejected') {
            currentState = yield* emitAndFold(currentState, {
              _tag: 'TurnEnd',
              result: { _tag: 'GateRejected', rejection: outcome.result.rejection },
            })
          }
        }
        break
      }

      case 'ProseChunk': {
        if (parseEvent.patternId === 'prose') {
          let id = proseMessage.get()
          if (!id) {
            id = proseMessage.create()
            proseMessage.set(id)
            currentState = yield* emitAndFold(currentState, {
              _tag: 'MessageStart',
              id,
              dest: config.defaultProseDest ?? 'user',
              artifactsRaw: null,
            })
          }
          currentState = yield* emitAndFold(currentState, {
            _tag: 'MessageChunk',
            id,
            text: parseEvent.text,
          })
          break
        }

        currentState = yield* emitAndFold(currentState, {
          _tag: 'ProseChunk',
          patternId: parseEvent.patternId,
          text: parseEvent.text,
        })
        break
      }

      case 'ProseEnd': {
        if (parseEvent.patternId === 'prose') {
          const id = proseMessage.get()
          if (id) {
            currentState = yield* emitAndFold(currentState, {
              _tag: 'MessageEnd',
              id,
            })
            proseMessage.set(null)
          }
          break
        }

        currentState = yield* emitAndFold(currentState, {
          _tag: 'ProseEnd',
          patternId: parseEvent.patternId,
          content: parseEvent.content,
          about: parseEvent.about,
        })
        break
      }

      case 'LensStart': {
        currentState = yield* emitAndFold(currentState, {
          _tag: 'LensStart',
          name: parseEvent.name,
        })
        break
      }

      case 'LensChunk': {
        currentState = yield* emitAndFold(currentState, {
          _tag: 'LensChunk',
          text: parseEvent.text,
        })
        break
      }

      case 'LensEnd': {
        currentState = yield* emitAndFold(currentState, {
          _tag: 'LensEnd',
          name: parseEvent.name,
          content: parseEvent.content,
        })
        break
      }


      case 'MessageTagOpen': {
        currentState = yield* emitAndFold(currentState, {
          _tag: 'MessageStart',
          id: parseEvent.id,
          dest: parseEvent.dest,
          artifactsRaw: parseEvent.artifactsRaw,
        })
        break
      }

      case 'MessageBodyChunk': {
        currentState = yield* emitAndFold(currentState, {
          _tag: 'MessageChunk',
          id: parseEvent.id,
          text: parseEvent.text,
        })
        break
      }

      case 'MessageTagClose': {
        currentState = yield* emitAndFold(currentState, {
          _tag: 'MessageEnd',
          id: parseEvent.id,
        })
        break
      }

      case 'TurnControl': {
        currentState = yield* emitAndFold(currentState, {
          _tag: 'TurnEnd',
          result: { _tag: 'Success', turnControl: parseEvent.decision },
        })
        break
      }

      case 'ParseError': {
        const error = parseEvent.error

        if (error._tag === 'UnclosedThink' || error._tag === 'UnclosedActions') {
          currentState = yield* emitAndFold(currentState, {
            _tag: 'StructuralParseError',
            error,
          })
          break
        }

        if (error._tag === 'TurnControlConflict') {
          currentState = yield* emitAndFold(currentState, {
            _tag: 'StructuralParseError',
            error,
          })
          break
        }

        // Tool-scoped error — needs toolCallId/tagName from the detail
        if (currentState.deadToolCalls.has(error.toolCallId)) break
        if (hasPriorOutcome(priorOutcomes, error.toolCallId)) break

        const registered = config.tools.get(error.tagName)
        if (registered) {
          const call = makeCallContext(error.toolCallId, error.tagName, registered)
          currentState = yield* emitAndFold(currentState, {
            _tag: 'ToolInputParseError',
            toolCallId: error.toolCallId,
            tagName: error.tagName,
            toolName: registered.tool.name,
            group: registered.groupName,
            error: { ...error, call },
          })
        }
        break
      }

      case 'ActionsOpen':
      case 'ActionsClose':
      case 'CommsOpen':
      case 'CommsClose':
        // Structural — no runtime events
        break
    }

    return currentState
  })
}

// =============================================================================
// Binding resolution helpers
// =============================================================================

function makeCallContext(toolCallId: string, tagName: string, registered: RegisteredTool): ToolCallContext {
  return {
    toolCallId,
    tagName,
    toolName: registered.tool.name,
    group: registered.groupName,
  }
}

function resolveBodyField(
  toolCallId: string,
  state: ReactorState,
  tools: ReadonlyMap<string, RegisteredTool>,
): string | undefined {
  const tagName = state.toolCallMap.get(toolCallId)
  if (!tagName) return undefined
  return tools.get(tagName)?.binding.body
}

interface ResolvedChildBinding {
  field: string
  bodyField?: string
  attributes?: readonly string[]
}

function resolveChildBinding(
  parentToolCallId: string,
  childTagName: string,
  state: ReactorState,
  tools: ReadonlyMap<string, RegisteredTool>,
): ResolvedChildBinding | undefined {
  const parentTagName = state.toolCallMap.get(parentToolCallId)
  if (!parentTagName) return undefined
  const registered = tools.get(parentTagName)
  if (!registered) return undefined
  const { binding } = registered

  if (binding.children) {
    for (const child of binding.children) {
      const tag = child.tag ?? child.field
      if (tag === childTagName) {
        return {
          field: child.field,
          bodyField: child.body,
          attributes: child.attributes,
        }
      }
    }
  }

  if (binding.childTags) {
    const ct = binding.childTags.find(ct => (ct.tag) === childTagName)
    if (ct) return { field: ct.field, bodyField: ct.field }
  }

  if (binding.childRecord?.tag === childTagName) {
    return {
      field: binding.childRecord.field,
      attributes: [binding.childRecord.keyAttr],
    }
  }

  return undefined
}
