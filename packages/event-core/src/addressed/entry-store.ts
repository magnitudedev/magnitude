import { Context, Effect, Layer, Option, Ref } from 'effect'
import { AddressedStoreError } from './errors'

export interface AddressedEntryStats {
  readonly storedBytes: number
}

export interface AddressedEntryStore {
  readonly load: (
    namespace: string,
    address: string
  ) => Effect.Effect<Option.Option<unknown>, AddressedStoreError>

  readonly stat: (
    namespace: string,
    address: string
  ) => Effect.Effect<Option.Option<AddressedEntryStats>, AddressedStoreError>

  readonly flush: (
    namespace: string,
    address: string,
    value: unknown
  ) => Effect.Effect<void, AddressedStoreError>
}

export const AddressedEntryStore = Context.GenericTag<AddressedEntryStore>('AddressedEntryStore')

const entryKey = (namespace: string, address: string) => `${namespace}\u0000${address}`

const cloneStoredValue = (value: unknown): unknown =>
  structuredClone(value)

export const makeInMemoryAddressedEntryStore = (
  initial?: Iterable<readonly [namespace: string, address: string, value: unknown]>
): Effect.Effect<AddressedEntryStore> =>
  Effect.gen(function* () {
    const entries = new Map<string, unknown>()
    for (const [namespace, address, value] of initial ?? []) {
      entries.set(entryKey(namespace, address), cloneStoredValue(value))
    }
    const ref = yield* Ref.make(entries)

    return {
      load: (namespace, address) =>
        Effect.map(
          Ref.get(ref),
          (map) => {
            const key = entryKey(namespace, address)
            return map.has(key)
              ? Option.some(cloneStoredValue(map.get(key)))
              : Option.none()
          }
        ),

      stat: (namespace, address) =>
        Effect.map(
          Ref.get(ref),
          (map) => {
            const key = entryKey(namespace, address)
            return map.has(key)
              ? Option.some({
                  storedBytes: estimateAddressedStoredBytes(map.get(key))
                })
              : Option.none()
          }
        ),

      flush: (namespace, address, value) =>
        Ref.update(ref, (map) => {
          const next = new Map(map)
          next.set(entryKey(namespace, address), cloneStoredValue(value))
          return next
        })
    }
  })

export const estimateAddressedStoredBytes = (value: unknown): number => {
  if (value === undefined) return 0
  try {
    return new TextEncoder().encode(JSON.stringify({ value })).byteLength
  } catch {
    return 0
  }
}

export const makeInMemoryAddressedEntryStoreLayer = (
  initial?: Iterable<readonly [namespace: string, address: string, value: unknown]>
): Layer.Layer<AddressedEntryStore> =>
  Layer.effect(AddressedEntryStore, makeInMemoryAddressedEntryStore(initial))
