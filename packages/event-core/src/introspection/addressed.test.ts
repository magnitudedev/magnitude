import { describe, expect, test } from 'bun:test'
import { Effect, Layer, Schema } from 'effect'
import { AddressedEntryStore, makeInMemoryAddressedEntryStore } from '../addressed/entry-store'
import { makeAddressSpaceRuntime } from '../addressed/address-space'
import { makeSchemaCodec } from '../addressed/codec'
import {
  AddressedIntrospectionRegistry,
  AddressedIntrospectionRegistryLive,
  makeAddressedSpaceIntrospection
} from './addressed'

const EntrySchema = Schema.Struct({
  text: Schema.String
})

const AddressedStoreLive = Layer.effect(
  AddressedEntryStore,
  makeInMemoryAddressedEntryStore()
)

describe('addressed introspection', () => {
  test('summarizes residents without retaining payloads', () => {
    const introspection = makeAddressedSpaceIntrospection('Test/messages', [
      {
        address: 'segment-a',
        dirty: true,
        pinCount: 1,
        pinOwners: ['display-view:test'],
        value: { text: 'hello' }
      },
      {
        address: 'segment-b',
        dirty: false,
        pinCount: 0,
        pinOwners: [],
        value: { text: 'stored' }
      }
    ])

    expect(introspection.namespace).toBe('Test/messages')
    expect(introspection.residentCount).toBe(2)
    expect(introspection.dirtyCount).toBe(1)
    expect(introspection.pinnedCount).toBe(1)
    expect(introspection.estimatedResidentBytes).toBeGreaterThan(0)
    expect(introspection.resident.map((resident) => resident.address)).toEqual(['segment-a', 'segment-b'])
    expect(introspection.resident[0]).not.toHaveProperty('value')
  })

  test('address spaces register only through the optional registry', async () => {
    const program = Effect.gen(function* () {
      const space = yield* makeAddressSpaceRuntime({
        namespace: 'Test/messages',
        codec: makeSchemaCodec(EntrySchema)
      })

      const mutation = space.makeMutation()
      yield* mutation.transaction.set('segment-a', { text: 'hello' })
      yield* mutation.commit
      yield* space.pin('display-view:test', ['segment-a'])

      const registry = yield* AddressedIntrospectionRegistry
      return yield* registry.current
    }).pipe(
      Effect.provide(
        Layer.mergeAll(AddressedStoreLive, AddressedIntrospectionRegistryLive)
      )
    )

    const introspections = await Effect.runPromise(program)
    expect(introspections).toHaveLength(1)
    expect(introspections[0].namespace).toBe('Test/messages')
    // The writer's pin shows up alongside the reader's — residency is
    // exactly the pinned set, and both interests are visible.
    expect(introspections[0].resident[0]).toMatchObject({
      address: 'segment-a',
      dirty: true,
      pinCount: 2,
      pinOwners: ['writer', 'display-view:test']
    })
  })

  test('address spaces expose stored entry stats through the registry', async () => {
    const program = Effect.gen(function* () {
      const store = yield* makeInMemoryAddressedEntryStore([
        ['Test/messages', 'segment-a', { text: 'stored' }]
      ])

      const space = yield* makeAddressSpaceRuntime({
        namespace: 'Test/messages',
        codec: makeSchemaCodec(EntrySchema)
      }).pipe(Effect.provideService(AddressedEntryStore, store))

      expect(space.namespace).toBe('Test/messages')

      const registry = yield* AddressedIntrospectionRegistry
      return yield* registry.stats('Test/messages', ['segment-a', 'missing'])
    }).pipe(Effect.provide(AddressedIntrospectionRegistryLive))

    const stats = await Effect.runPromise(program)
    expect(stats).toHaveLength(1)
    expect(stats[0]).toMatchObject({
      address: 'segment-a'
    })
    expect(stats[0].storedBytes).toBeGreaterThan(0)
  })

  test('address spaces run normally without the registry service', async () => {
    const program = Effect.gen(function* () {
      const space = yield* makeAddressSpaceRuntime({
        namespace: 'Test/messages',
        codec: makeSchemaCodec(EntrySchema)
      })
      const mutation = space.makeMutation()
      yield* mutation.transaction.set('segment-a', { text: 'hello' })
      return yield* mutation.commit
    }).pipe(Effect.provide(AddressedStoreLive))

    const changed = await Effect.runPromise(program)
    expect([...changed]).toEqual(['segment-a'])
  })
})
