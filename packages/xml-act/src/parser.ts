import { createStackMachine } from './machine'
import { createTokenizer } from './tokenizer'
import type { Format, XmlActFrame, XmlActEvent, ToolDef } from './format/types'
import { createCurrentFormat } from './format/index'
import type { TagSchema } from './execution/binding-validator'
import { createId } from './util'
import { classifyXmlActEvent, createCoalescingLayer, mergeXmlActEvent } from './coalescing'

export type IdGenerator = () => string
export const defaultIdGenerator: IdGenerator = createId

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

  const onEvent = (event: E) => {
    if (config.filter && !config.filter(event)) return
    if (coalescingLayer) {
      coalescingLayer.accept(event)
      return
    }
    rawEmit(event)
  }

  const machine = createStackMachine<F, E>(config.initialFrame, onEvent)

  const tokenizer = createTokenizer((signal) => {
    if (machine.done) return

    switch (signal.type) {
      case 'open': {
        const result = config.format.resolve(signal.tagName, machine.stack)
        if (result._tag === 'handle') {
          machine.apply(result.handler.open({
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
          machine.apply(result.handler.close({
            tagName: signal.tagName,
            afterNewline: signal.afterNewline,
            stack: machine.stack,
          }))
        } else {
          machine.apply(config.format.onUnknownClose(
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
          machine.apply(result.handler.selfClose({
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
          machine.apply(config.format.onContent(top, signal.text))
        }
        break
      }
    }
  }, config.knownToolTags)

  const processChunk = (chunk: string): readonly E[] => {
    const start = events.length
    if (!machine.done) tokenizer.push(chunk)
    if (coalescingLayer) coalescingLayer.flush()
    return events.slice(start)
  }

  const flush = (): readonly E[] => {
    const start = events.length
    tokenizer.end()
    machine.apply(config.format.onFlush(machine.stack))
    if (coalescingLayer) coalescingLayer.flush()
    return events.slice(start)
  }

  return {
    push(chunk) {
      if (!machine.done) tokenizer.push(chunk)
    },
    end() {
      tokenizer.end()
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
): StreamingParser {
  const tools: ToolDef[] = [...(knownTags ?? [])].map((tag) => ({
    tag,
    childTags: childTagMap?.get(tag) ?? new Set(),
    schema: tagSchemas?.get(tag),
  }))

  const { format, structuralTags } = createCurrentFormat(tools, defaultMessageDest)

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
