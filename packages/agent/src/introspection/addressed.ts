import { Effect, Option } from 'effect'
import { Introspection } from '@magnitudedev/event-core'

export type AddressedSpaceIntrospection = Introspection.AddressedSpaceIntrospection
export type AddressedResidentIntrospection = Introspection.AddressedResidentIntrospection
export type AddressedStoredEntryIntrospection = Introspection.AddressedStoredEntryIntrospection

export const currentAddressedSpaceIntrospection: Effect.Effect<readonly AddressedSpaceIntrospection[]> =
  Effect.gen(function* () {
    const registry = yield* Effect.serviceOption(Introspection.AddressedIntrospectionRegistry)
    if (Option.isNone(registry)) return []
    return yield* registry.value.current
  })

export const currentAddressedEntryStats = (
  namespace: string,
  addresses: Iterable<string>
): Effect.Effect<readonly AddressedStoredEntryIntrospection[]> =>
  Effect.gen(function* () {
    const registry = yield* Effect.serviceOption(Introspection.AddressedIntrospectionRegistry)
    if (Option.isNone(registry)) return []
    return yield* registry.value.stats(namespace, addresses)
  })
