/**
 * Parser — context-aware XML turn parser.
 *
 * Each token is routed through resolveOpenHandler / resolveCloseHandler /
 * resolveSelfCloseHandler in resolve.ts. Resolution returns typed bound handlers
 * that capture the narrowed parent frame. No as-casts in the dispatch path.
 *
 * Emits ToolParseError and StructuralParseError events.
 * FilterReady is surfaced via an onFilterReady callback (not in TurnEngineEvent).
 */

import { createStackMachine } from '../machine'
import { deriveParameters } from '../engine/parameter-schema'
import type {
  TurnEngineEvent,
  FilterReady,
} from '../types'
import type { Frame } from './types'
import type { ParserConfig } from './types'
import type { InvokeContext } from './handler-context'
import type { HandlerContext } from './handler-context'
import { classifyEvent, mergeEvent, type CoalescingBuffer } from './coalesce'
import { endTopProse } from './content'
import { flushAllFrames } from './flush'
import { pushToken, type ParserLoopContext } from './dispatch'
import { resolveCloseHandler } from './resolve'
import type { ParserOp } from './ops'

export type { ParserConfig } from './types'

// =============================================================================
// Parser
// =============================================================================

export interface XmlActParser {
  pushToken(token: import('../types').Token): void
  end(): void
  drain(): readonly TurnEngineEvent[]
}

export function createParser(
  config: ParserConfig,
  onFilterReady?: (event: FilterReady) => void,
): XmlActParser {
  let idCounter = 0
  const generateId = config.generateId ?? (() => `xact-${++idCounter}-${Date.now().toString(36)}`)

  // Derive tool schemas eagerly
  const toolSchemas = new Map<string, ReturnType<typeof deriveParameters>>()
  for (const [tagName, registeredTool] of config.tools) {
    try {
      const schema = deriveParameters(registeredTool.tool.inputSchema.ast)
      toolSchemas.set(tagName, schema)
    } catch {
      // schema derivation failed — tool parameters won't be validated
    }
  }

  const events: TurnEngineEvent[] = []
  let coalescingBuffer: CoalescingBuffer | null = null

  function flushCoalescing(): void {
    if (coalescingBuffer === null) return
    events.push(coalescingBuffer.event as TurnEngineEvent)
    coalescingBuffer = null
  }

  function emit(event: TurnEngineEvent): void {
    const key = classifyEvent(event)
    if (key === null) {
      flushCoalescing()
      events.push(event)
      return
    }
    if (coalescingBuffer !== null && coalescingBuffer.key === key) {
      mergeEvent(coalescingBuffer.event, event)
      return
    }
    flushCoalescing()
    coalescingBuffer = { key, event: { ...event } as CoalescingBuffer['event'] }
  }

  const machine = createStackMachine<Frame, TurnEngineEvent>(
    { type: 'prose', body: '', pendingNewlines: 0, hasContent: false },
    emit,
  )

  // InvokeContext — tool registry and schema access for invoke-related handlers
  const invokeCtx: InvokeContext = {
    tools: config.tools,
    toolSchemas,
    onFilterReady,
  }

  // HandlerContext — minimal context passed to all handlers
  const handlerCtx: HandlerContext = {
    generateId,
    invokeCtx,
  }

  // Deferred yield state
  const deferredYield = {
    target: null as 'user' | 'invoke' | 'worker' | 'parent' | null,
    postYieldHasContent: false,
  }

  // ParserLoopContext — wires machine + handlerCtx + deferredYield for pushToken
  const loopCtx: ParserLoopContext = {
    machine,
    handlerCtx,
    deferredYield,
    pendingCloseStack: [],
    escapeDepth: 0,
  }

  function end(): void {
    if (machine.mode === 'done') return

    // EOF confirms any tentative close stack
    if (loopCtx.pendingCloseStack.length > 0) {
      for (const pc of loopCtx.pendingCloseStack) {
        const top = machine.peek()
        if (top) {
          const handler = resolveCloseHandler(pc.tagName, top)
          if (handler) {
            machine.apply(handler.close(handlerCtx))
          }
        }
      }
      loopCtx.pendingCloseStack = []
    }

    flushCoalescing()

    if (deferredYield.target !== null) {
      const termination = deferredYield.postYieldHasContent ? 'runaway' : 'natural'
      events.push({
        _tag: 'TurnEnd',
        result: { _tag: 'Success', turnControl: { target: deferredYield.target }, termination },
      })
      deferredYield.target = null
      machine.apply([{ type: 'done' }])
      return
    }

    flushAllFrames(
      () => machine.peek(),
      (ops) => machine.apply(ops),
      invokeCtx,
    )
  }

  return {
    pushToken: (token) => pushToken(token, loopCtx),
    end,
    drain(): readonly TurnEngineEvent[] {
      flushCoalescing()
      const pending = [...events]
      events.length = 0
      return pending
    },
  }
}
