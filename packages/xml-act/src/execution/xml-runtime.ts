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
import type { ToolContext } from '@magnitudedev/tools'

import { createStreamingXmlParser, defaultIdGenerator, type IdGenerator } from '../parser'
import { createShortId } from '../util'
import type { ParseEvent, ParsedElement } from '../format/types'
import type {
  XmlRuntimeConfig,
  XmlRuntimeEvent,
  ToolInterceptor,

  RegisteredTool,
  ReactorState,
} from '../types'
import { XmlRuntimeCrash, ToolInterceptorTag } from '../types'
import { dispatchTool, type DispatchContext, type DispatchResult } from './tool-dispatcher'
import { buildInput } from './input-builder'
import { observeOutput } from '../output-query'
import { validateBinding, type TagSchema } from './binding-validator'
import { initialReactorState, foldReactorState } from './reactor-state'
import { coerceAttributeValue } from '../format/coerce'

// =============================================================================
// Sentinel for end-of-stream
// =============================================================================

const END = Symbol('END')
type QueueItem = XmlRuntimeEvent | typeof END

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
            if (reg.binding.attributes) reg.binding.attributes.forEach(a => valid.add(a.attr))
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
          // Reactor glue
          // ---------------------------------------------------------------

          function emitAndFold(
            state: ReactorState,
            event: XmlRuntimeEvent,
          ): Effect.Effect<ReactorState> {
            return Effect.gen(function* () {
              yield* Queue.offer(queue, event)
              return foldReactorState(state, event)
            })
          }

          const createMessageId = createShortId
          let activeProseMessageId: string | null = null
          const emittedFields = new Map<string, Set<string>>()
          const pendingAttrNormalizationChildren = new Map<string, Set<number>>()

          function react(
            state: ReactorState,
            parseEvent: ParseEvent,
          ): Effect.Effect<ReactorState> {
            return reactImpl(
              state, parseEvent, emitAndFold,
              config, tagSchemas, Option.getOrUndefined(interceptor),
              priorToolCallIds, priorOutcomes, queue,
              {
                get: () => activeProseMessageId,
                set: (id) => { activeProseMessageId = id },
                create: createMessageId,
              },
              emittedFields,
              pendingAttrNormalizationChildren,
            )
          }

          // Producer fiber: parse → react → dispatch, offer events to queue.
          // Each LLM chunk is passed to parser.processChunk() which returns
          // coalesced events (text events merged by the parser's coalescing layer).
          const producer = Effect.gen(function* () {
            yield* xmlStream.pipe(
              Stream.mapError((e) => new XmlRuntimeCrash(e.message, e)),
              Stream.runForEach((chunk) =>
                Effect.gen(function* () {
                  let state = yield* Ref.get(stateRef)

                  // After TurnEnd, stop consuming. Parser observing mode
                  // handles termination classification before TurnEnd is emitted.
                  if (state.stopped) {
                    return yield* Effect.fail(new XmlRuntimeCrash('__runaway_abort__'))
                  }

                  const parseEvents = parser.processChunk(chunk)
                  for (const pe of parseEvents) {
                    if (state.stopped) break
                    state = yield* react(state, pe)
                  }
                  yield* Ref.set(stateRef, state)
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
              yield* Ref.set(stateRef, state)
            }

            if (!state.stopped) {
              yield* Queue.offer(queue, {
                _tag: 'TurnEnd',
                result: { _tag: 'Success', turnControl: null, termination: 'natural' },
              } satisfies XmlRuntimeEvent)
            }

            yield* Queue.offer(queue, END)
          }).pipe(
            // Catch typed errors (XmlRuntimeCrash)
            Effect.catchAll((crash) =>
              Effect.gen(function* () {
                if (crash instanceof XmlRuntimeCrash) {
                  // Sentinel for runaway abort — TurnEnd already emitted by parser/reactor
                  if (crash.message === '__runaway_abort__') {
                    yield* Queue.offer(queue, END)
                    return
                  }
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
                const message = describeDefect(defect)
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
  queue: Queue.Queue<QueueItem>,
  proseMessage: {
    readonly get: () => string | null
    readonly set: (id: string | null) => void
    readonly create: () => string
  },
  emittedFields: Map<string, Set<string>>,
  pendingAttrNormalizationChildren: Map<string, Set<number>>,
): Effect.Effect<ReactorState> {
  return Effect.gen(function* () {
    let currentState = state
    const cleanupToolCallState = (toolCallId: string) => {
      emittedFields.delete(toolCallId)
      pendingAttrNormalizationChildren.delete(toolCallId)
    }

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

        const callEmitted = emittedFields.get(parseEvent.toolCallId) ?? new Set<string>()
        emittedFields.set(parseEvent.toolCallId, callEmitted)

        // Emit canonical streaming events from attributes:
        // - canonical attr -> ToolInputFieldValue
        // - attr->childTag swap -> ToolInputChildStarted + ToolInputChildComplete
        for (const [attrName, attrValue] of parseEvent.attributes) {
          const boundAttr = registered.binding.attributes?.find(a => a.attr === attrName)
          if (boundAttr) {
            if (!callEmitted.has(boundAttr.field)) {
              currentState = yield* emitAndFold(currentState, {
                _tag: 'ToolInputFieldValue',
                toolCallId: parseEvent.toolCallId,
                field: boundAttr.field,
                value: attrValue,
              })
              callEmitted.add(boundAttr.field)
            }
            continue
          }

          const boundChild = registered.binding.childTags?.find(ct => ct.tag === attrName)
          if (boundChild) {
            if (!callEmitted.has(boundChild.field)) {
              currentState = yield* emitAndFold(currentState, {
                _tag: 'ToolInputChildStarted',
                toolCallId: parseEvent.toolCallId,
                field: boundChild.field,
                index: 0,
                attributes: {},
              })
              currentState = yield* emitAndFold(currentState, {
                _tag: 'ToolInputChildComplete',
                toolCallId: parseEvent.toolCallId,
                field: boundChild.field,
                index: 0,
                value: { [boundChild.field]: String(attrValue) },
              })
              callEmitted.add(boundChild.field)
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
          if (!registered) break

          currentState = yield* emitAndFold(currentState, {
            _tag: 'ToolInputParseError',
            toolCallId: parseEvent.toolCallId,
            tagName: tagNameForBody,
            toolName: registered.tool.name,
            group: registered.groupName,
            error: {
              _tag: 'UnexpectedBody',
              id: parseEvent.toolCallId,
              tagName: tagNameForBody,
              detail: `Tool <${tagNameForBody}> does not accept body content`,
            },
          })
        }
        break
      }

      case 'ChildOpened': {
        if (currentState.deadToolCalls.has(parseEvent.parentToolCallId)) break
        if (hasPriorOutcome(priorOutcomes, parseEvent.parentToolCallId)) break
        if (isInFlight(priorToolCallIds, priorOutcomes, parseEvent.parentToolCallId)) break

        const parentTagName = currentState.toolCallMap.get(parseEvent.parentToolCallId)
        if (!parentTagName) break
        const registered = config.tools.get(parentTagName)
        if (!registered) break
        const attrSpec = registered.binding.attributes?.find(a => a.attr === parseEvent.childTagName)

        if (attrSpec) {
          const pending = pendingAttrNormalizationChildren.get(parseEvent.parentToolCallId) ?? new Set<number>()
          pending.add(parseEvent.childIndex)
          pendingAttrNormalizationChildren.set(parseEvent.parentToolCallId, pending)
          break
        }

        const childInfo = resolveChildBinding(parseEvent.parentToolCallId, parseEvent.childTagName, currentState, config.tools)
        if (!childInfo) break

        const callEmitted = emittedFields.get(parseEvent.parentToolCallId) ?? new Set<string>()
        emittedFields.set(parseEvent.parentToolCallId, callEmitted)
        if (callEmitted.has(childInfo.field)) break

        const attrRecord: Record<string, string | number | boolean> = {}
        for (const [k, v] of parseEvent.attributes) attrRecord[k] = v

        currentState = yield* emitAndFold(currentState, {
          _tag: 'ToolInputChildStarted',
          toolCallId: parseEvent.parentToolCallId,
          field: childInfo.field,
          index: parseEvent.childIndex,
          attributes: attrRecord,
        })
        break
      }

      case 'ChildBodyChunk': {
        if (currentState.deadToolCalls.has(parseEvent.parentToolCallId)) break
        if (hasPriorOutcome(priorOutcomes, parseEvent.parentToolCallId)) break
        if (isInFlight(priorToolCallIds, priorOutcomes, parseEvent.parentToolCallId)) break

        const pending = pendingAttrNormalizationChildren.get(parseEvent.parentToolCallId)
        if (pending?.has(parseEvent.childIndex)) break

        const childInfo = resolveChildBinding(parseEvent.parentToolCallId, parseEvent.childTagName, currentState, config.tools)
        if (childInfo?.bodyField) {
          const callEmitted = emittedFields.get(parseEvent.parentToolCallId)
          if (callEmitted?.has(childInfo.field)) break
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

        const parentTagName = currentState.toolCallMap.get(parseEvent.parentToolCallId)
        if (!parentTagName) break
        const registered = config.tools.get(parentTagName)
        if (!registered) break

        const callEmitted = emittedFields.get(parseEvent.parentToolCallId) ?? new Set<string>()
        emittedFields.set(parseEvent.parentToolCallId, callEmitted)

        const pending = pendingAttrNormalizationChildren.get(parseEvent.parentToolCallId)
        const isPendingAttrNormalization = pending?.has(parseEvent.childIndex) ?? false
        if (isPendingAttrNormalization) {
          pending?.delete(parseEvent.childIndex)
          if (pending && pending.size === 0) pendingAttrNormalizationChildren.delete(parseEvent.parentToolCallId)

          const attrSpec = registered.binding.attributes?.find(a => a.attr === parseEvent.childTagName)
          const simpleText = parseEvent.attributes.size === 0
          if (attrSpec && simpleText && !callEmitted.has(attrSpec.field)) {
            const trimmedBody = parseEvent.body.trim()
            const attrSchema = tagSchemas.get(parentTagName)?.attributes.get(attrSpec.attr)
            if (attrSchema) {
              const coerced = coerceAttributeValue(trimmedBody, attrSchema.type)
              if (!coerced.ok) {
                currentState = yield* emitAndFold(currentState, {
                  _tag: 'ToolInputParseError',
                  toolCallId: parseEvent.parentToolCallId,
                  tagName: parentTagName,
                  toolName: registered.tool.name,
                  group: registered.groupName,
                  error: {
                    _tag: 'InvalidAttributeValue',
                    id: parseEvent.parentToolCallId,
                    tagName: parentTagName,
                    attribute: attrSpec.attr,
                    expected: attrSchema.type,
                    received: trimmedBody,
                    detail: `Invalid value for attribute '${attrSpec.attr}' on <${parentTagName}>: "${trimmedBody}"`,
                  },
                })
                break
              }
              currentState = yield* emitAndFold(currentState, {
                _tag: 'ToolInputFieldValue',
                toolCallId: parseEvent.parentToolCallId,
                field: attrSpec.field,
                value: coerced.value,
              })
              callEmitted.add(attrSpec.field)
              break
            }

            currentState = yield* emitAndFold(currentState, {
              _tag: 'ToolInputFieldValue',
              toolCallId: parseEvent.parentToolCallId,
              field: attrSpec.field,
              value: trimmedBody,
            })
            callEmitted.add(attrSpec.field)
          }
          break
        }

        const childInfo = resolveChildBinding(parseEvent.parentToolCallId, parseEvent.childTagName, currentState, config.tools)
        if (!childInfo) break
        if (callEmitted.has(childInfo.field)) break

        const value: Record<string, unknown> = {}
        if (childInfo.attributes) {
          for (const attrSpec of childInfo.attributes) {
            const v = parseEvent.attributes.get(attrSpec.attr)
            if (v !== undefined) value[attrSpec.field] = v
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
        callEmitted.add(childInfo.field)
        break
      }

      case 'TagClosed': {
        // Replay: outcome known → suppress entirely
        if (hasPriorOutcome(priorOutcomes, parseEvent.toolCallId)) {
          cleanupToolCallState(parseEvent.toolCallId)
          break
        }
        // Dead tool calls: skip dispatch
        if (currentState.deadToolCalls.has(parseEvent.toolCallId)) {
          cleanupToolCallState(parseEvent.toolCallId)
          break
        }

        const registered = config.tools.get(parseEvent.tagName)
        if (!registered) {
          cleanupToolCallState(parseEvent.toolCallId)
          break
        }

        const tagSchema = tagSchemas.get(parseEvent.tagName)
        const canonicalElement = tagSchema
          ? canonicalizeAttrChildTagSwaps(parseEvent.element, registered.binding, tagSchema)
          : parseEvent.element
        const input = buildInput(canonicalElement, registered.binding)

        // The dispatcher emits events via this callback — state stays in sync
        // automatically because every event goes through emitAndFold.
        const dispatchCtx: DispatchContext = {
          tools: config.tools,
          interceptor,
          toolContext: {
            emit: (value: unknown) => Queue.offer(queue, {
              _tag: 'ToolEmission',
              toolCallId: parseEvent.toolCallId,
              value,
            } as XmlRuntimeEvent),
          } satisfies ToolContext<unknown>,
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
          currentState = yield* emitAndFold(currentState, {
            _tag: 'ToolInputReady',
            toolCallId: parseEvent.toolCallId,
            input,
          })
        }

        // Dispatch tool — events emitted via callback, state folded automatically
        const result: DispatchResult = yield* dispatchTool(canonicalElement, dispatchCtx)

        if (result._tag === 'ParseError') {
          currentState = yield* emitAndFold(currentState, {
            _tag: 'ToolInputParseError',
            toolCallId: parseEvent.toolCallId,
            tagName: parseEvent.tagName,
            toolName: registered.tool.name,
            group: registered.groupName,
            error: result.error,
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
        cleanupToolCallState(parseEvent.toolCallId)
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
              to: null,
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

      case 'MessageStart': {
        currentState = yield* emitAndFold(currentState, {
          _tag: 'MessageStart',
          id: parseEvent.id,
          to: parseEvent.to,
        })
        break
      }

      case 'MessageChunk': {
        currentState = yield* emitAndFold(currentState, {
          _tag: 'MessageChunk',
          id: parseEvent.id,
          text: parseEvent.text,
        })
        break
      }

      case 'MessageEnd': {
        currentState = yield* emitAndFold(currentState, {
          _tag: 'MessageEnd',
          id: parseEvent.id,
        })
        break
      }

      case 'TurnControl': {
        currentState = yield* emitAndFold(currentState, {
          _tag: 'TurnEnd',
          result: parseEvent.decision === 'finish'
            ? { _tag: 'Success', turnControl: 'finish', evidence: parseEvent.evidence, termination: parseEvent.termination }
            : { _tag: 'Success', turnControl: parseEvent.decision, termination: parseEvent.termination },
        })
        break
      }

      case 'ParseError': {
        const error = parseEvent.error

        if (error._tag === 'UnclosedThink') {
          currentState = yield* emitAndFold(currentState, {
            _tag: 'StructuralParseError',
            error,
          })
          break
        }

        if (error._tag === 'TurnControlConflict' || error._tag === 'FinishWithoutEvidence') {
          currentState = yield* emitAndFold(currentState, {
            _tag: 'StructuralParseError',
            error,
          })
          break
        }

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
        break
      }


    }

    return currentState
  })
}

// =============================================================================
// Binding resolution helpers
// =============================================================================

function resolveBodyField(
  toolCallId: string,
  state: ReactorState,
  tools: ReadonlyMap<string, RegisteredTool>,
): string | undefined {
  const tagName = state.toolCallMap.get(toolCallId)
  if (!tagName) return undefined
  return tools.get(tagName)?.binding.body
}

function canonicalizeAttrChildTagSwaps(
  element: ParsedElement,
  binding: RegisteredTool['binding'],
  tagSchema: TagSchema,
): ParsedElement {
  let attributes = new Map(element.attributes)
  let children = [...element.children]

  if (binding.attributes) {
    for (const attrBinding of binding.attributes) {
      const hasCanonicalAttr = attributes.has(attrBinding.attr)
      if (hasCanonicalAttr) continue
      const matchingChildren = children.filter(
        (child) => child.tagName === attrBinding.attr && child.attributes.size === 0,
      )
      if (matchingChildren.length !== 1) continue
      const child = matchingChildren[0]
      const trimmedBody = child.body.trim()
      const attrType = tagSchema.attributes.get(attrBinding.attr)?.type
      if (attrType) {
        const coerced = coerceAttributeValue(trimmedBody, attrType)
        attributes.set(attrBinding.attr, coerced.ok ? coerced.value : trimmedBody)
      } else {
        attributes.set(attrBinding.attr, trimmedBody)
      }
      let removedOne = false
      children = children.filter((c) => {
        if (!removedOne && c === child) {
          removedOne = true
          return false
        }
        return true
      })
    }
  }

  if (binding.childTags) {
    for (const childBinding of binding.childTags) {
      const hasCanonicalChild = children.some((child) => child.tagName === childBinding.tag)
      if (hasCanonicalChild) continue
      const attrValue = attributes.get(childBinding.tag)
      if (attrValue === undefined) continue
      children = [...children, {
        tagName: childBinding.tag,
        attributes: new Map(),
        body: String(attrValue),
      }]
      attributes.delete(childBinding.tag)
    }
  }

  return {
    tagName: element.tagName,
    toolCallId: element.toolCallId,
    attributes,
    body: element.body,
    children,
  }
}

interface ResolvedChildBinding {
  field: string
  bodyField?: string
  attributes?: readonly { readonly field: string; readonly attr: string }[]
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
      attributes: [{ field: binding.childRecord.keyAttr, attr: binding.childRecord.keyAttr }],
    }
  }

  return undefined
}
