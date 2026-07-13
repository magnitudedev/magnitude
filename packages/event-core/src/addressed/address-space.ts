import { Effect, Option, SynchronizedRef } from 'effect'
import type { AddressedEntryCodec } from './codec'
import { AddressedEntryStore } from './entry-store'
import type { AddressedError } from './errors'
import { registerAddressedSpaceIntrospection } from '../introspection/addressed'

/**
 * Residency is exactly the pinned set. No cache policy, no capacity.
 *
 * - Readers pin what they read: the framework replaces each consumer's pins
 *   with its tracked reads immediately after each handler. Pins fault their
 *   targets in; a pin target the index vouches for is resident-or-stored by
 *   construction, so a missing target is an integrity defect, not a
 *   recoverable error.
 * - The writer pins what it wrote: each write transaction's commit replaces
 *   the writer owner's pins with the transaction's written addresses, so
 *   streaming producers rewrite the same resident segment every commit.
 *   Transactions that stage nothing don't rotate the writer pin.
 * - When an entry's last pin releases, it is flushed if dirty, then dropped.
 *   Dirty entries are always pinned (writer until rotation, readers while
 *   displayed); `flushDirty` (snapshot capture) flushes them in place.
 * - Reads load from the store when not resident and retain the loaded value
 *   as resident. Retained residency becomes durable when a pin lands before
 *   the next release boundary; otherwise it drops there.
 */
interface ResidentEntry<Value> {
  readonly value: Value
  readonly dirty: boolean
  readonly pinCount: number
}

type ResidentTable<Value> = ReadonlyMap<string, ResidentEntry<Value>>
type PinOwner = string

/** The pin owner used by write-transaction commits. */
const WRITER_PIN_OWNER = 'writer'

interface AddressSpaceState<Value> {
  readonly resident: ResidentTable<Value>
  readonly pins: ReadonlyMap<PinOwner, ReadonlySet<string>>
}

export interface AddressSpaceTransaction<Value> {
  readonly get: (address: string) => Effect.Effect<Option.Option<Value>, AddressedError>
  readonly set: (address: string, value: Value) => Effect.Effect<void>
  /**
   * Stage an address into the commit's changed set without writing an entry.
   * Used for collection sentinels: structural index changes (append, remove,
   * member add/remove) are observable changes even when no existing entry
   * content was rewritten.
   */
  readonly markChanged: (address: string) => void
}

export interface AddressSpaceMutation<Value> {
  readonly transaction: AddressSpaceTransaction<Value>
  readonly commit: Effect.Effect<ReadonlySet<string>, AddressedError>
}

export interface AddressSpaceTransactionResult<A> {
  readonly value: A
  readonly changed: ReadonlySet<string>
}

export interface AddressSpaceRuntime<Value> {
  readonly namespace: string
  readonly get: (address: string) => Effect.Effect<Option.Option<Value>, AddressedError>
  readonly makeMutation: () => AddressSpaceMutation<Value>
  readonly transact: <A, E, R>(
    body: (transaction: AddressSpaceTransaction<Value>) => Effect.Effect<A, E, R>
  ) => Effect.Effect<AddressSpaceTransactionResult<A>, E | AddressedError, R>
  readonly pin: (
    owner: string,
    addresses: Iterable<string>
  ) => Effect.Effect<void, AddressedError>
  readonly unpin: (owner: string) => Effect.Effect<void, AddressedError>
  readonly flushDirty: Effect.Effect<void, AddressedError>
  readonly reset: Effect.Effect<void>
}

export interface AddressSpaceRuntimeOptions<Value> {
  readonly namespace: string
  readonly codec: AddressedEntryCodec<Value>
}

const toAddressSet = (addresses: Iterable<string>): ReadonlySet<string> =>
  new Set(addresses)

export const makeAddressSpaceRuntime = <Value>(
  options: AddressSpaceRuntimeOptions<Value>
): Effect.Effect<AddressSpaceRuntime<Value>, never, AddressedEntryStore> =>
  Effect.gen(function* () {
    const store = yield* AddressedEntryStore
    const stateRef = yield* SynchronizedRef.make<AddressSpaceState<Value>>({
      resident: new Map(),
      pins: new Map()
    })
    const contextFor = (address: string) => ({
      namespace: options.namespace,
      address
    })

    const integrityDefect = (address: string, violation: string): never => {
      throw new Error(
        `AddressSpace(${options.namespace}): ${violation} for address "${address}"`
      )
    }

    const withPinCount = (
      resident: ResidentTable<Value>,
      address: string,
      delta: 1 | -1
    ): ResidentTable<Value> => {
      const entry = resident.get(address)
      if (!entry) return integrityDefect(address, 'pin bookkeeping on non-resident entry')
      const next = new Map(resident)
      next.set(address, { ...entry, pinCount: entry.pinCount + delta })
      return next
    }

    const flushEntry = (address: string, entry: ResidentEntry<Value>) =>
      Effect.flatMap(
        options.codec.encode(entry.value, contextFor(address)),
        (encoded) => store.flush(options.namespace, address, encoded)
      )

    /**
     * Restore `resident = pinned`: flush-then-drop every entry whose last pin
     * is gone. Runs at every boundary that can strand an unpinned entry.
     */
    const releaseUnpinned = (
      state: AddressSpaceState<Value>
    ): Effect.Effect<AddressSpaceState<Value>, AddressedError> =>
      Effect.gen(function* () {
        let resident: Map<string, ResidentEntry<Value>> | undefined
        for (const [address, entry] of state.resident) {
          if (entry.pinCount > 0) continue
          if (entry.dirty) {
            yield* flushEntry(address, entry)
          }
          resident = resident ?? new Map(state.resident)
          resident.delete(address)
        }
        return resident ? { ...state, resident } : state
      })

    const loadValue = (
      state: AddressSpaceState<Value>,
      address: string
    ): Effect.Effect<Option.Option<Value>, AddressedError> => {
      const existing = state.resident.get(address)
      if (existing) return Effect.succeed(Option.some(existing.value))

      return Effect.flatMap(
        store.load(options.namespace, address),
        (encoded) =>
          Option.isNone(encoded)
            ? Effect.succeed(Option.none<Value>())
            : Effect.map(
                options.codec.decode(encoded.value, contextFor(address)),
                Option.some
              )
      )
    }

    /**
     * Read a value, loading from the store when not resident. The loaded
     * value is retained as resident (unpinned) — it becomes durable residency
     * when a pin lands before the next settle boundary (the framework pins a
     * consumer's tracked reads immediately after each handler), and is
     * otherwise dropped at that boundary.
     */
    const get = (address: string): Effect.Effect<Option.Option<Value>, AddressedError> =>
      SynchronizedRef.modifyEffect(
        stateRef,
        (state) =>
          Effect.map(
            loadValue(state, address),
            (value) => {
              if (Option.isNone(value) || state.resident.has(address)) {
                return [value, state] as const
              }
              const resident = new Map(state.resident)
              resident.set(address, { value: value.value, dirty: false, pinCount: 0 })
              return [value, { ...state, resident }] as const
            }
          )
      )

    const replaceOwnerPins = (
      state: AddressSpaceState<Value>,
      owner: PinOwner,
      nextAddresses: ReadonlySet<string>
    ): AddressSpaceState<Value> => {
      const currentAddresses = state.pins.get(owner) ?? new Set<string>()

      let resident = state.resident
      for (const address of nextAddresses) {
        if (!currentAddresses.has(address)) resident = withPinCount(resident, address, 1)
      }
      for (const address of currentAddresses) {
        if (!nextAddresses.has(address)) resident = withPinCount(resident, address, -1)
      }

      const pins = new Map(state.pins)
      if (nextAddresses.size === 0) {
        pins.delete(owner)
      } else {
        pins.set(owner, nextAddresses)
      }

      return { ...state, resident, pins }
    }

    const faultPinTargets = (
      state: AddressSpaceState<Value>,
      owner: PinOwner,
      addresses: ReadonlySet<string>
    ): Effect.Effect<AddressSpaceState<Value>, AddressedError> =>
      Effect.gen(function* () {
        const currentAddresses = state.pins.get(owner) ?? new Set<string>()
        let resident: Map<string, ResidentEntry<Value>> | undefined
        for (const address of addresses) {
          if (currentAddresses.has(address)) continue
          if ((resident ?? state.resident).has(address)) continue
          const value = yield* loadValue(state, address)
          if (Option.isNone(value)) {
            return integrityDefect(address, 'pin target is neither resident nor stored')
          }
          resident = resident ?? new Map(state.resident)
          resident.set(address, { value: value.value, dirty: false, pinCount: 0 })
        }
        return resident ? { ...state, resident } : state
      })

    const applyOwnerPins = (
      state: AddressSpaceState<Value>,
      owner: PinOwner,
      addresses: ReadonlySet<string>
    ): Effect.Effect<AddressSpaceState<Value>, AddressedError> =>
      Effect.gen(function* () {
        const faulted = yield* faultPinTargets(state, owner, addresses)
        const repinned = replaceOwnerPins(faulted, owner, addresses)
        return yield* releaseUnpinned(repinned)
      })

    const pin = (
      owner: PinOwner,
      addresses: Iterable<string>
    ): Effect.Effect<void, AddressedError> => {
      const nextAddresses = toAddressSet(addresses)
      return SynchronizedRef.modifyEffect(
        stateRef,
        (state) =>
          Effect.map(
            applyOwnerPins(state, owner, nextAddresses),
            (nextState) => [undefined, nextState] as const
          )
      )
    }

    const unpin = (owner: PinOwner): Effect.Effect<void, AddressedError> =>
      SynchronizedRef.modifyEffect(
        stateRef,
        (state) =>
          state.pins.has(owner)
            ? Effect.map(
                applyOwnerPins(state, owner, new Set()),
                (nextState) => [undefined, nextState] as const
              )
            : Effect.succeed([undefined, state] as const)
      )

    const makeMutation = (): AddressSpaceMutation<Value> => {
      const stagedWrites = new Map<string, Value>()
      const stagedMarks = new Set<string>()
      let committed = false

      const transaction: AddressSpaceTransaction<Value> = {
        get: (address) => {
          // Staged values are never undefined; absence means the address was
          // not written in this transaction.
          const staged = stagedWrites.get(address)
          return staged === undefined
            ? get(address)
            : Effect.succeed(Option.some(staged))
        },
        set: (address, value) =>
          Effect.sync(() => {
            stagedWrites.set(address, value)
          }),
        markChanged: (address) => {
          stagedMarks.add(address)
        }
      }

      const commit: Effect.Effect<ReadonlySet<string>, AddressedError> = Effect.suspend(() => {
        if (committed) return Effect.succeed<ReadonlySet<string>>(new Set())
        committed = true
        // Only transactions that write rotate the writer pin. A transaction
        // with no writes leaves every entry's state untouched — marks-only
        // transactions just report their sentinels as changed.
        if (stagedWrites.size === 0) {
          const changed: ReadonlySet<string> = new Set(stagedMarks)
          stagedMarks.clear()
          return Effect.succeed(changed)
        }
        return SynchronizedRef.modifyEffect(
          stateRef,
          (state) =>
            Effect.gen(function* () {
              const changed = new Set<string>(stagedMarks)
              const resident = new Map(state.resident)
              const written = new Set<string>()
              for (const [address, value] of stagedWrites) {
                const existing = resident.get(address)
                resident.set(address, {
                  value,
                  dirty: true,
                  pinCount: existing?.pinCount ?? 0
                })
                changed.add(address)
                written.add(address)
              }

              // Rotate the writer pin to this transaction's writes (all just
              // written, hence resident — no fault-in), then flush-and-drop
              // whatever the rotation unpinned.
              const repinned = replaceOwnerPins({ ...state, resident }, WRITER_PIN_OWNER, written)
              const settled = yield* releaseUnpinned(repinned)
              stagedWrites.clear()
              stagedMarks.clear()
              return [changed, settled] as const
            })
        )
      })

      return { transaction, commit }
    }

    const transact = <A, E, R>(
      body: (transaction: AddressSpaceTransaction<Value>) => Effect.Effect<A, E, R>
    ): Effect.Effect<AddressSpaceTransactionResult<A>, E | AddressedError, R> =>
      Effect.gen(function* () {
        const mutation = makeMutation()
        const value = yield* body(mutation.transaction)
        const changed = yield* mutation.commit
        return { value, changed }
      })

    const flushDirty = SynchronizedRef.modifyEffect(
      stateRef,
      (state) =>
        Effect.gen(function* () {
          let resident: Map<string, ResidentEntry<Value>> | undefined
          for (const [address, entry] of state.resident) {
            if (!entry.dirty) continue
            yield* flushEntry(address, entry)
            resident = resident ?? new Map(state.resident)
            resident.set(address, { ...entry, dirty: false })
          }
          return [undefined, resident ? { ...state, resident } : state] as const
        })
    )

    const reset = SynchronizedRef.set(stateRef, {
      resident: new Map<string, ResidentEntry<Value>>(),
      pins: new Map<PinOwner, ReadonlySet<string>>()
    })

    yield* registerAddressedSpaceIntrospection(
      options.namespace,
      SynchronizedRef.get(stateRef),
      (address) => store.stat(options.namespace, address)
    )

    return {
      namespace: options.namespace,
      get,
      makeMutation,
      transact,
      pin,
      unpin,
      flushDirty,
      reset
    }
  })
