import { Addressed } from '@magnitudedev/event-core'
import { Effect, Option, Ref } from 'effect'

const keyFor = (namespace: string, address: string) => `${namespace}\u0000${address}`

export const makeCountingAddressedEntryStore = Effect.gen(function* () {
  const entriesRef = yield* Ref.make(new Map<string, unknown>())
  const loadsRef = yield* Ref.make(new Map<string, number>())

  const store: Addressed.AddressedEntryStore = {
    load: (namespace, address) =>
      Effect.gen(function* () {
        yield* Ref.update(loadsRef, (loads) => {
          const next = new Map(loads)
          const key = keyFor(namespace, address)
          next.set(key, (next.get(key) ?? 0) + 1)
          return next
        })
        const entries = yield* Ref.get(entriesRef)
        const key = keyFor(namespace, address)
        return entries.has(key)
          ? Option.some(entries.get(key))
          : Option.none()
      }),
    stat: (namespace, address) =>
      Effect.map(Ref.get(entriesRef), (entries) => {
        const key = keyFor(namespace, address)
        return entries.has(key)
          ? Option.some({
              storedBytes: Addressed.estimateAddressedStoredBytes(entries.get(key))
            })
          : Option.none()
      }),
    flush: (namespace, address, value) =>
      Ref.update(entriesRef, (entries) => {
        const next = new Map(entries)
        next.set(keyFor(namespace, address), value)
        return next
      })
  }

  const loadCount = (namespace: string, address: string) =>
    Effect.map(Ref.get(loadsRef), (loads) => loads.get(keyFor(namespace, address)) ?? 0)

  return { store, loadCount } as const
})
