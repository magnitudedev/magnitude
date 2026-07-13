import { describe, expect, test } from 'bun:test'
import { Effect, Option } from 'effect'
import { makeInMemoryAddressedEntryStore } from '../entry-store'

const mutateFirstItemText = (value: unknown, text: string): void => {
  if (typeof value !== 'object' || value === null) return

  const items = Reflect.get(value, 'items')
  if (!Array.isArray(items)) return

  const first = items[0]
  if (typeof first !== 'object' || first === null) return

  Reflect.set(first, 'text', text)
}

describe('in-memory addressed entry store', () => {
  test('loads absence and isolates identical addresses by namespace', async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const store = yield* makeInMemoryAddressedEntryStore([
        ['namespace-a', 'same-address', { value: 'a' }],
        ['namespace-b', 'same-address', { value: 'b' }]
      ])

      expect(yield* store.load('namespace-a', 'same-address'))
        .toEqual(Option.some({ value: 'a' }))
      expect(yield* store.load('namespace-b', 'same-address'))
        .toEqual(Option.some({ value: 'b' }))
      expect(Option.isNone(yield* store.load('namespace-a', 'missing'))).toBe(true)

      yield* store.flush('namespace-a', 'same-address', { value: 'next-a' })

      expect(yield* store.load('namespace-a', 'same-address'))
        .toEqual(Option.some({ value: 'next-a' }))
      expect(yield* store.load('namespace-b', 'same-address'))
        .toEqual(Option.some({ value: 'b' }))
    }))
  })

  test('copies values at initial load, flush, and read boundaries', async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const initial = { items: [{ text: 'initial' }] }
      const store = yield* makeInMemoryAddressedEntryStore([
        ['namespace', 'initial', initial]
      ])

      initial.items[0].text = 'mutated after construction'
      expect(yield* store.load('namespace', 'initial'))
        .toEqual(Option.some({ items: [{ text: 'initial' }] }))

      const flushed = { items: [{ text: 'flushed' }] }
      yield* store.flush('namespace', 'flushed', flushed)
      flushed.items[0].text = 'mutated after flush'
      expect(yield* store.load('namespace', 'flushed'))
        .toEqual(Option.some({ items: [{ text: 'flushed' }] }))

      const loaded = yield* store.load('namespace', 'flushed')
      if (Option.isSome(loaded)) {
        mutateFirstItemText(loaded.value, 'mutated loaded value')
      }

      expect(yield* store.load('namespace', 'flushed'))
        .toEqual(Option.some({ items: [{ text: 'flushed' }] }))
    }))
  })
})
