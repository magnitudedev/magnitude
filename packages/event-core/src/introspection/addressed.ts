import { Context, Effect, Layer, Option, Ref } from 'effect'
import type { AddressedEntryStats } from '../addressed/entry-store'
import type { AddressedStoreError } from '../addressed/errors'

export interface AddressedResidentIntrospection {
  readonly address: string
  readonly dirty: boolean
  readonly pinCount: number
  readonly pinOwners: readonly string[]
  readonly estimatedBytes: number
}

export interface AddressedStoredEntryIntrospection {
  readonly address: string
  readonly storedBytes: number
}

export interface AddressedSpaceIntrospection {
  readonly namespace: string
  readonly resident: readonly AddressedResidentIntrospection[]
  readonly residentCount: number
  readonly dirtyCount: number
  readonly pinnedCount: number
  readonly estimatedResidentBytes: number
  readonly estimatedDirtyBytes: number
}

export interface AddressedSpaceIntrospectionInput {
  readonly address: string
  readonly dirty: boolean
  readonly pinCount: number
  readonly pinOwners: readonly string[]
  readonly value: unknown
}

export interface AddressedResidentState<Value> {
  readonly value: Value
  readonly dirty: boolean
  readonly pinCount: number
}

export interface AddressedSpaceState<Value> {
  readonly resident: ReadonlyMap<string, AddressedResidentState<Value>>
  readonly pins: ReadonlyMap<string | symbol, ReadonlySet<string>>
}

export interface AddressedSpaceInspector {
  readonly namespace: string
  readonly current: Effect.Effect<AddressedSpaceIntrospection>
  readonly stats: (
    addresses: Iterable<string>
  ) => Effect.Effect<readonly AddressedStoredEntryIntrospection[]>
}

export interface AddressedIntrospectionRegistryService {
  readonly register: (inspector: AddressedSpaceInspector) => Effect.Effect<void>
  readonly current: Effect.Effect<readonly AddressedSpaceIntrospection[]>
  readonly stats: (
    namespace: string,
    addresses: Iterable<string>
  ) => Effect.Effect<readonly AddressedStoredEntryIntrospection[]>
}

export class AddressedIntrospectionRegistry extends Context.Tag('AddressedIntrospectionRegistry')<
  AddressedIntrospectionRegistry,
  AddressedIntrospectionRegistryService
>() {}

const estimateBytes = (value: unknown): number => {
  if (value === undefined) return 0
  try {
    return new TextEncoder().encode(JSON.stringify(value)).byteLength
  } catch {
    return 0
  }
}

const pinOwnerLabel = (owner: string | symbol): string =>
  typeof owner === 'symbol'
    ? owner.description ?? String(owner)
    : owner

const pinOwnersFor = (
  pins: ReadonlyMap<string | symbol, ReadonlySet<string>>,
  address: string
): readonly string[] => {
  const owners: string[] = []
  for (const [owner, addresses] of pins) {
    if (addresses.has(address)) owners.push(pinOwnerLabel(owner))
  }
  return owners
}

export const makeAddressedSpaceIntrospection = (
  namespace: string,
  entries: Iterable<AddressedSpaceIntrospectionInput>
): AddressedSpaceIntrospection => {
  const resident: AddressedResidentIntrospection[] = []
  let dirtyCount = 0
  let pinnedCount = 0
  let estimatedResidentBytes = 0
  let estimatedDirtyBytes = 0

  for (const entry of entries) {
    const estimatedEntryBytes = estimateBytes(entry.value)
    if (entry.dirty) {
      dirtyCount += 1
      estimatedDirtyBytes += estimatedEntryBytes
    }
    if (entry.pinCount > 0) pinnedCount += 1
    estimatedResidentBytes += estimatedEntryBytes

    resident.push({
      address: entry.address,
      dirty: entry.dirty,
      pinCount: entry.pinCount,
      pinOwners: entry.pinOwners,
      estimatedBytes: estimatedEntryBytes
    })
  }

  resident.sort((a, b) => a.address.localeCompare(b.address))

  return {
    namespace,
    resident,
    residentCount: resident.length,
    dirtyCount,
    pinnedCount,
    estimatedResidentBytes,
    estimatedDirtyBytes
  }
}

export const AddressedIntrospectionRegistryLive: Layer.Layer<AddressedIntrospectionRegistry> =
  Layer.effect(
    AddressedIntrospectionRegistry,
    Effect.gen(function* () {
      const inspectorsRef = yield* Ref.make(new Map<string, AddressedSpaceInspector>())

      return {
        register: (inspector) =>
          Ref.update(inspectorsRef, (inspectors) => {
            const next = new Map(inspectors)
            next.set(inspector.namespace, inspector)
            return next
          }),

        current: Effect.gen(function* () {
          const inspectors = [...(yield* Ref.get(inspectorsRef)).values()]
          return yield* Effect.forEach(inspectors, (inspector) => inspector.current)
        }),

        stats: (namespace, addresses) =>
          Effect.gen(function* () {
            const inspector = (yield* Ref.get(inspectorsRef)).get(namespace)
            if (!inspector) return []
            return yield* inspector.stats(addresses)
          })
      } satisfies AddressedIntrospectionRegistryService
    })
  )

export const registerAddressedSpaceIntrospection = <Value>(
  namespace: string,
  readState: Effect.Effect<AddressedSpaceState<Value>>,
  statEntry: (
    address: string
  ) => Effect.Effect<Option.Option<AddressedEntryStats>, AddressedStoreError>
): Effect.Effect<void> =>
  Effect.gen(function* () {
    const registry = yield* Effect.serviceOption(AddressedIntrospectionRegistry)
    if (Option.isNone(registry)) return

    yield* registry.value.register({
      namespace,
      current: Effect.map(readState, (state) =>
        makeAddressedSpaceIntrospection(
          namespace,
          [...state.resident].map(([address, entry]) => ({
            address,
            dirty: entry.dirty,
            pinCount: entry.pinCount,
            pinOwners: pinOwnersFor(state.pins, address),
            value: entry.value
          }))
        )
      ),
      stats: (addresses) =>
        Effect.map(
          Effect.forEach(
            [...new Set(addresses)],
            (address) =>
              Effect.map(statEntry(address), (stats) =>
                Option.isSome(stats)
                  ? {
                      address,
                      storedBytes: stats.value.storedBytes
                    } satisfies AddressedStoredEntryIntrospection
                  : null
              ),
            { concurrency: 'unbounded' }
          ),
          (stats) => stats.filter((stat): stat is AddressedStoredEntryIntrospection => stat !== null)
        ).pipe(
          Effect.catchAll(() =>
            Effect.succeed([] as readonly AddressedStoredEntryIntrospection[])
          )
        )
    })
  })
