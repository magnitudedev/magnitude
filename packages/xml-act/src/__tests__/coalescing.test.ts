import { describe, expect, it } from 'bun:test'
import { createCoalescingLayer } from '../coalescing'

type TestEvent =
  | { _tag: 'Text'; key: string; text: string }
  | { _tag: 'Mark'; value: string }

describe('createCoalescingLayer', () => {
  it('merges consecutive same-key text events', () => {
    const emitted: TestEvent[] = []
    const layer = createCoalescingLayer<TestEvent>({
      emit(event) {
        emitted.push(event)
      },
      classify(event) {
        return event._tag === 'Text' ? event.key : null
      },
      merge(target, source) {
        if (target._tag === 'Text' && source._tag === 'Text') {
          target.text += source.text
        }
      },
    })

    layer.accept({ _tag: 'Text', key: 'a', text: 'hel' })
    layer.accept({ _tag: 'Text', key: 'a', text: 'lo' })
    layer.flush()

    expect(emitted).toEqual([{ _tag: 'Text', key: 'a', text: 'hello' }])
  })

  it('flushes on key change and buffers new key', () => {
    const emitted: TestEvent[] = []
    const layer = createCoalescingLayer<TestEvent>({
      emit(event) {
        emitted.push(event)
      },
      classify(event) {
        return event._tag === 'Text' ? event.key : null
      },
      merge(target, source) {
        if (target._tag === 'Text' && source._tag === 'Text') {
          target.text += source.text
        }
      },
    })

    layer.accept({ _tag: 'Text', key: 'a', text: 'one' })
    layer.accept({ _tag: 'Text', key: 'b', text: 'two' })

    expect(emitted).toEqual([{ _tag: 'Text', key: 'a', text: 'one' }])

    layer.flush()
    expect(emitted).toEqual([
      { _tag: 'Text', key: 'a', text: 'one' },
      { _tag: 'Text', key: 'b', text: 'two' },
    ])
  })

  it('non-text flushes buffered text then emits immediately', () => {
    const emitted: TestEvent[] = []
    const layer = createCoalescingLayer<TestEvent>({
      emit(event) {
        emitted.push(event)
      },
      classify(event) {
        return event._tag === 'Text' ? event.key : null
      },
      merge(target, source) {
        if (target._tag === 'Text' && source._tag === 'Text') {
          target.text += source.text
        }
      },
    })

    layer.accept({ _tag: 'Text', key: 'a', text: 'x' })
    layer.accept({ _tag: 'Mark', value: '!' })

    expect(emitted).toEqual([{ _tag: 'Text', key: 'a', text: 'x' }, { _tag: 'Mark', value: '!' }])
  })

  it('flush emits buffered event', () => {
    const emitted: TestEvent[] = []
    const layer = createCoalescingLayer<TestEvent>({
      emit(event) {
        emitted.push(event)
      },
      classify(event) {
        return event._tag === 'Text' ? event.key : null
      },
      merge(target, source) {
        if (target._tag === 'Text' && source._tag === 'Text') {
          target.text += source.text
        }
      },
    })

    layer.accept({ _tag: 'Text', key: 'a', text: 'x' })
    layer.flush()

    expect(emitted).toEqual([{ _tag: 'Text', key: 'a', text: 'x' }])
  })

  it('flush on empty buffer is no-op', () => {
    const emitted: TestEvent[] = []
    const layer = createCoalescingLayer<TestEvent>({
      emit(event) {
        emitted.push(event)
      },
      classify(event) {
        return event._tag === 'Text' ? event.key : null
      },
      merge(target, source) {
        if (target._tag === 'Text' && source._tag === 'Text') {
          target.text += source.text
        }
      },
    })

    layer.flush()
    expect(emitted).toEqual([])
  })

  it('handles interleaved text/non-text/text', () => {
    const emitted: TestEvent[] = []
    const layer = createCoalescingLayer<TestEvent>({
      emit(event) {
        emitted.push(event)
      },
      classify(event) {
        return event._tag === 'Text' ? event.key : null
      },
      merge(target, source) {
        if (target._tag === 'Text' && source._tag === 'Text') {
          target.text += source.text
        }
      },
    })

    layer.accept({ _tag: 'Text', key: 'a', text: 'hi' })
    layer.accept({ _tag: 'Mark', value: '-' })
    layer.accept({ _tag: 'Text', key: 'a', text: 'there' })
    layer.flush()

    expect(emitted).toEqual([
      { _tag: 'Text', key: 'a', text: 'hi' },
      { _tag: 'Mark', value: '-' },
      { _tag: 'Text', key: 'a', text: 'there' },
    ])
  })
})
