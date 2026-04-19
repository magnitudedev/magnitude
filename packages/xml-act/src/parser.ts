import { createStackMachine, type Op } from './machine'
import { createTokenizer } from './tokenizer'
import type { Format, XmlActFrame, XmlActEvent, ToolDef } from './format/types'
import { createCurrentFormat } from './format/index'
import type { TagSchema } from './execution/binding-validator'
import { createId } from './util'
import { classifyXmlActEvent, createCoalescingLayer, mergeXmlActEvent } from './coalescing'

export type IdGenerator = () => string
export const defaultIdGenerator: IdGenerator = createId

type ObserveOutcome = 'pending' | 'runaway'

/**
 * Observer for content after a yield tag.
 * Since yield tags are self-closing and the stop sequence halts generation,
 * any non-whitespace content after yield indicates a runaway.
 */
class PostYieldObserver {
  private buffer = ''

  feed(chunk: string): ObserveOutcome {
    this.buffer += chunk
    return this.process(false)
  }

  finish(): 'natural' | 'runaway' {
    const outcome = this.process(true)
    return outcome === 'runaway' ? 'runaway' : 'natural'
  }

  private process(isEof: boolean): ObserveOutcome {
    // Trim leading whitespace
    const trimmedLeading = this.buffer.replace(/^\s+/u, '')
    if (trimmedLeading !== this.buffer) {
      this.buffer = trimmedLeading
    }

    if (this.buffer.length === 0) return 'pending'

    // Any non-whitespace content after yield is a runaway
    return 'runaway'
  }
}

function isTurnControlEvent(event: unknown): event is {
  _tag: 'TurnControl'
  target: 'user' | 'tool' | 'worker' | 'parent'
  termination: 'natural' | 'runaway'
} {
  if (!event || typeof event !== 'object') return false
  const candidate = event as { _tag?: unknown; target?: unknown }
  return candidate._tag === 'TurnControl'
    && typeof candidate.target === 'string'
    && ['user', 'tool', 'worker', 'parent'].includes(candidate.target)
}

export interface StreamingParser {
  push(chunk: string): void
  end(): void
  processChunk(chunk: string): readonly XmlActEvent[]
  flush(): readonly XmlActEvent[]
  readonly events: readonly XmlActEvent[]
}

export function createParser<F extends { readonly type: string }, E>(config: {
  format: Format<F, E>
  initialFrame: F
  knownToolTags?: ReadonlySet<string>
  generateId?: () => string
  emit?: (event: E) => void
  filter?: (event: E) => boolean
  coalesce?: {
    classify: (event: E) => string | null
    merge: (target: E, source: E) => void
  }
}): {
  push(chunk: string): void
  end(): void
  processChunk(chunk: string): readonly E[]
  flush(): readonly E[]
  readonly events: readonly E[]
} {
  const events: E[] = []
  const generateId = config.generateId ?? createId
  const userEmit = config.emit ?? (() => {})

  const rawEmit = (event: E) => {
    events.push(event)
    userEmit(event)
  }

  const coalescingLayer = config.coalesce
    ? createCoalescingLayer<E>({
        emit: rawEmit,
        classify: config.coalesce.classify,
        merge: config.coalesce.merge,
      })
    : null

  const routeEvent = (event: E) => {
    if (config.filter && !config.filter(event)) return
    if (coalescingLayer) {
      coalescingLayer.accept(event)
      return
    }
    rawEmit(event)
  }

  let activeBatch: E[] | null = null
  const onEvent = (event: E) => {
    if (activeBatch) {
      activeBatch.push(event)
      return
    }
    routeEvent(event)
  }

  const machine = createStackMachine<F, E>(config.initialFrame, onEvent)
  let deferredTurnControl: E | null = null
  let observer: PostYieldObserver | null = null

  const emitDeferredTurnControl = (termination: 'natural' | 'runaway') => {
    if (!deferredTurnControl || !isTurnControlEvent(deferredTurnControl)) return
    routeEvent({
      ...deferredTurnControl,
      termination,
    } as E)
    deferredTurnControl = null
  }

  const applyOps = (ops: ReadonlyArray<Op<F, E>>) => {
    activeBatch = []
    machine.apply(ops)
    const batch = activeBatch
    activeBatch = null

    if (machine.mode === 'observing' && !observer) observer = new PostYieldObserver()

    if (machine.mode === 'observing' && !deferredTurnControl) {
      for (let i = batch.length - 1; i >= 0; i--) {
        if (isTurnControlEvent(batch[i])) {
          deferredTurnControl = batch[i]
          batch.splice(i, 1)
          break
        }
      }
    }

    for (const event of batch) routeEvent(event)
  }

  const tokenizer = createTokenizer((signal) => {
    if (machine.mode !== 'active') return

    switch (signal.type) {
      case 'open': {
        const result = config.format.resolve(signal.tagName, machine.stack)
        if (result._tag === 'handle') {
          applyOps(result.handler.open({
            tagName: signal.tagName,
            attrs: signal.attrs,
            afterNewline: signal.afterNewline,
            stack: machine.stack,
            generateId,
          }))
        } else {
          machine.apply(config.format.onUnknownOpen(
            signal.tagName, signal.attrs, signal.afterNewline, machine.stack, signal.raw ?? `<${signal.tagName}>`,
          ))
        }
        break
      }
      case 'close': {
        const result = config.format.resolve(signal.tagName, machine.stack)
        if (result._tag === 'handle') {
          applyOps(result.handler.close({
            tagName: signal.tagName,
            afterNewline: signal.afterNewline,
            stack: machine.stack,
          }))
        } else {
          applyOps(config.format.onUnknownClose(
            signal.tagName,
            machine.stack,
            signal.raw ?? `</${signal.tagName}>`,
          ))
        }
        break
      }
      case 'selfClose': {
        const result = config.format.resolve(signal.tagName, machine.stack)
        if (result._tag === 'handle') {
          applyOps(result.handler.selfClose({
            tagName: signal.tagName,
            attrs: signal.attrs,
            afterNewline: signal.afterNewline,
            stack: machine.stack,
            generateId,
          }))
        } else {
          machine.apply(config.format.onUnknownOpen(
            signal.tagName, signal.attrs, signal.afterNewline, machine.stack, signal.raw ?? `<${signal.tagName}>`,
          ))
        }
        break
      }
      case 'content': {
        const top = machine.peek()
        if (top) {
          applyOps(config.format.onContent(top, signal.text))
        }
        break
      }
    }
  }, config.knownToolTags)

  const processChunk = (chunk: string): readonly E[] => {
    const start = events.length

    if (machine.mode === 'observing') {
      if (observer?.feed(chunk) === 'runaway') {
        emitDeferredTurnControl('runaway')
        machine.finalize()
      }
    } else if (machine.mode === 'active') {
      tokenizer.push(chunk)
    }

    if (coalescingLayer) coalescingLayer.flush()
    return events.slice(start)
  }

  const flush = (): readonly E[] => {
    const start = events.length

    if (machine.mode === 'observing') {
      const termination = observer?.finish() ?? 'natural'
      emitDeferredTurnControl(termination)
      machine.finalize()
    } else {
      tokenizer.end()
      applyOps(config.format.onFlush(machine.stack))
    }

    if (coalescingLayer) coalescingLayer.flush()
    return events.slice(start)
  }

  return {
    push(chunk) {
      if (machine.mode === 'observing') {
        if (observer?.feed(chunk) === 'runaway') {
          emitDeferredTurnControl('runaway')
          machine.finalize()
        }
        if (coalescingLayer) coalescingLayer.flush()
        return
      }
      if (machine.mode === 'active') tokenizer.push(chunk)
    },
    end() {
      if (machine.mode === 'observing') {
        const termination = observer?.finish() ?? 'natural'
        emitDeferredTurnControl(termination)
        machine.finalize()
      } else {
        tokenizer.end()
      }
      if (coalescingLayer) coalescingLayer.flush()
    },
    processChunk,
    flush,
    get events() {
      return events
    },
  }
}

export function createStreamingXmlParser(
  knownTags?: ReadonlySet<string>,
  childTagMap?: ReadonlyMap<string, ReadonlySet<string>>,
  tagSchemas?: ReadonlyMap<string, TagSchema>,
  generateId?: () => string,
  defaultMessageDest?: string,
  yieldTags?: ReadonlyArray<string>,
): StreamingParser {
  const tools: ToolDef[] = [...(knownTags ?? [])].map((tag) => ({
    tag,
    childTags: childTagMap?.get(tag) ?? new Set(),
    schema: tagSchemas?.get(tag),
  }))

  const { format, structuralTags } = createCurrentFormat(tools, defaultMessageDest, yieldTags)

  let emittedTurnControl = false

  return createParser<XmlActFrame, XmlActEvent>({
    format,
    initialFrame: { type: 'prose', body: '', pendingNewlines: 0, tags: structuralTags },
    knownToolTags: knownTags,
    generateId,
    filter(event) {
      if (event._tag !== 'TurnControl') return true
      if (emittedTurnControl) return false
      emittedTurnControl = true
      return true
    },
    coalesce: {
      classify: classifyXmlActEvent,
      merge: mergeXmlActEvent,
    },
  })
}
