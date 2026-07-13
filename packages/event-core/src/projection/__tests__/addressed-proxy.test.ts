import { describe, expect, test } from 'bun:test'
import { Effect, Schema } from 'effect'
import {
  AddressedEntryStore,
  makeInMemoryAddressedEntryStore
} from '../../addressed/entry-store'
import * as ProjectionAddressed from '../addressed'
import type { AddressedSequenceIndex } from '../../addressed/collections/sequence'

const ItemSchema = Schema.Struct({
  id: Schema.String,
  text: Schema.String
})
type Item = Schema.Schema.Type<typeof ItemSchema>

/**
 * Build a committed sequence and a Proxy view over it, with a recorder that
 * captures every address the Proxy touches.
 */
const makeProxyFixture = (count: number) =>
  Effect.gen(function* () {
    const store = yield* makeInMemoryAddressedEntryStore()
    const descriptor = ProjectionAddressed.sequence(ItemSchema)
    const runtime = yield* descriptor.makeRuntime('ProxyTest/items').pipe(
      Effect.provideService(AddressedEntryStore, store)
    )

    const mutation = runtime.makeMutation()
    const handle = descriptor.makeHandle('ProxyTest/items', runtime, mutation.transaction)
    let index: AddressedSequenceIndex = handle.empty
    for (let i = 0; i < count; i += 1) {
      index = yield* handle.append(index, { id: `m${i}`, text: `text ${i}` })
    }
    yield* mutation.commit

    const consumer = descriptor.makeConsumer('ProxyTest/items', runtime)
    const recorded = new Set<string>()
    const proxy = ProjectionAddressed.makeSequenceProxy<Item>(
      index,
      consumer,
      (address) => recorded.add(address)
    )
    return { proxy, recorded, sentinel: consumer.sentinelAddress, index }
  })

describe('addressed sequence proxy', () => {
  test('acts like a readonly array: length, index, at, slice, iteration', async () => {
    const { proxy } = await Effect.runPromise(makeProxyFixture(5))

    expect(proxy.length).toBe(5)
    expect(proxy[0]).toEqual({ id: 'm0', text: 'text 0' })
    expect(proxy[4]).toEqual({ id: 'm4', text: 'text 4' })
    expect(proxy[5]).toBeUndefined()
    expect(proxy.at(-1)).toEqual({ id: 'm4', text: 'text 4' })
    expect(proxy.slice(-2).map((item) => item.id)).toEqual(['m3', 'm4'])
    expect(proxy.slice(1, 3).map((item) => item.id)).toEqual(['m1', 'm2'])
    expect([...proxy].map((item) => item.id)).toEqual(['m0', 'm1', 'm2', 'm3', 'm4'])
  })

  test('whole-array methods delegate with native semantics', async () => {
    const { proxy } = await Effect.runPromise(makeProxyFixture(4))

    expect(proxy.map((item) => item.id)).toEqual(['m0', 'm1', 'm2', 'm3'])
    expect(proxy.filter((item) => item.id !== 'm1')).toHaveLength(3)
    expect(proxy.find((item) => item.id === 'm2')).toEqual({ id: 'm2', text: 'text 2' })
    expect(proxy.findIndex((item) => item.id === 'm3')).toBe(3)
    expect(proxy.some((item) => item.id === 'm0')).toBe(true)
    expect(proxy.every((item) => item.text.startsWith('text'))).toBe(true)
    expect(proxy.reduce((acc, item) => acc + item.id.length, 0)).toBe(8)
    expect(proxy.join('|')).toContain('object')

    // includes/indexOf use SameValueZero — a structurally equal object is a
    // different value; the actual element is found.
    const second = proxy[1]
    expect(proxy.includes(second as Item)).toBe(true)
    expect(proxy.includes({ id: 'm1', text: 'text 1' })).toBe(false)
    expect(proxy.indexOf(second as Item)).toBe(1)
  })

  test('satisfies object protocol invariants: Object.keys, spread, descriptors', async () => {
    const { proxy } = await Effect.runPromise(makeProxyFixture(3))

    expect(Object.keys(proxy)).toEqual(['0', '1', '2'])
    expect(Object.getOwnPropertyDescriptor(proxy, 'length')).toMatchObject({ value: 3 })
    const spread = { ...proxy }
    expect(Object.keys(spread)).toEqual(['0', '1', '2'])
    expect(spread[0]).toEqual({ id: 'm0', text: 'text 0' })
    expect(spread[2]).toEqual({ id: 'm2', text: 'text 2' })
    expect(Array.isArray(proxy)).toBe(true)
    expect(0 in proxy).toBe(true)
    expect(2 in proxy).toBe(true)
    expect(3 in proxy).toBe(false)
    expect('slice' in proxy).toBe(true)
  })

  test('unknown property probes resolve without reading content', async () => {
    const { proxy, recorded, sentinel } = await Effect.runPromise(makeProxyFixture(3))

    expect(Reflect.get(proxy, 'then')).toBeUndefined()
    expect(Reflect.get(proxy, 'push')).toBeUndefined()
    expect(Reflect.get(proxy, 'someRandomProperty')).toBeUndefined()

    // Probes record the sentinel (structure was consulted) but never touch
    // segment content.
    expect(recorded).toEqual(new Set([sentinel]))
  })

  test('records the sentinel and exactly the touched segment addresses', async () => {
    const { proxy, recorded, sentinel, index } = await Effect.runPromise(makeProxyFixture(3))

    void proxy.length
    expect(recorded).toEqual(new Set([sentinel]))

    void proxy[0]
    expect(recorded).toEqual(new Set([sentinel, index.segments[0]!.address]))
  })
})
