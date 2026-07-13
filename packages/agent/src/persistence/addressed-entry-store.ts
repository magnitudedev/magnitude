import { Effect, Layer, Option } from 'effect'
import { Addressed } from '@magnitudedev/event-core'
import type { MagnitudeStorageShape } from '@magnitudedev/storage'
import { ChatPersistence } from './chat-persistence-service'

const toAddressedStoreError =
  (
    operation: 'load' | 'flush' | 'stat',
    namespace: string,
    address: string
  ) =>
  (cause: unknown) =>
    new Addressed.AddressedStoreError({
      operation,
      namespace,
      address,
      cause
    })

export const makeChatAddressedEntryStoreLayer = (
  storage: MagnitudeStorageShape,
  sessionId?: string
): Layer.Layer<Addressed.AddressedEntryStore, never, ChatPersistence> =>
  Layer.effect(
    Addressed.AddressedEntryStore,
    Effect.gen(function* () {
      const persistence = yield* ChatPersistence
      const resolvedSessionId = sessionId
        ? Effect.succeed(sessionId)
        : yield* Effect.cached(
          persistence.getSessionMetadata().pipe(
            Effect.map((metadata) => metadata.sessionId)
          )
        )

      const sessionIdFor = (
        operation: 'load' | 'flush' | 'stat',
        namespace: string,
        address: string
      ) =>
        resolvedSessionId.pipe(
          Effect.mapError(toAddressedStoreError(operation, namespace, address))
        )

      return {
        load: (namespace, address) =>
          Effect.gen(function* () {
            const resolved = yield* sessionIdFor('load', namespace, address)
            const entry = yield* storage.sessions
              .readAddressedEntry(resolved, namespace, address)
              .pipe(Effect.mapError(toAddressedStoreError('load', namespace, address)))
            return entry === null ? Option.none() : Option.some(entry.value)
          }),

        stat: (namespace, address) =>
          Effect.gen(function* () {
            const resolved = yield* sessionIdFor('stat', namespace, address)
            const stats = yield* storage.sessions
              .statAddressedEntry(resolved, namespace, address)
              .pipe(Effect.mapError(toAddressedStoreError('stat', namespace, address)))
            return stats === null ? Option.none() : Option.some(stats)
          }),

        flush: (namespace, address, value) =>
          Effect.gen(function* () {
            const resolved = yield* sessionIdFor('flush', namespace, address)
            yield* storage.sessions
              .writeAddressedEntry(resolved, namespace, address, value)
              .pipe(Effect.mapError(toAddressedStoreError('flush', namespace, address)))
          })
      } satisfies Addressed.AddressedEntryStore
    })
  )
