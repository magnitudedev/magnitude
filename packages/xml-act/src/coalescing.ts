import type { XmlActEvent } from './format/types'

type Writable<T> = { -readonly [K in keyof T]: T[K] }

export interface CoalescingLayer<E> {
  accept(event: E): void
  flush(): void
}

export function createCoalescingLayer<E>(config: {
  emit: (event: E) => void
  classify: (event: E) => string | null
  merge: (target: Writable<E>, source: E) => void
}): CoalescingLayer<E> {
  let buffer: { key: string; event: Writable<E> } | null = null

  const flush = () => {
    if (buffer === null) return
    config.emit(buffer.event)
    buffer = null
  }

  const accept = (event: E) => {
    const key = config.classify(event)
    if (key === null) {
      flush()
      config.emit(event)
      return
    }

    if (buffer !== null && buffer.key === key) {
      config.merge(buffer.event, event)
      return
    }

    flush()
    buffer = { key, event: { ...event } }
  }

  return { accept, flush }
}

export function classifyXmlActEvent(event: XmlActEvent): string | null {
  switch (event._tag) {
    case 'LensChunk':
      return 'lens'
    case 'MessageChunk':
      return `message:${event.id}`
    case 'BodyChunk':
      return `body:${event.toolCallId}`
    case 'ChildBodyChunk':
      return `child:${event.parentToolCallId}:${event.childTagName}:${event.childIndex}`
    default:
      return null
  }
}

/**
 * Merge two coalesced text events by appending source text to target.
 * Target is writable because the coalescing layer owns buffered events.
 */
export function mergeXmlActEvent(target: Writable<XmlActEvent>, source: XmlActEvent): void {
  if ('text' in target && 'text' in source) {
    target.text += source.text
  }
}
