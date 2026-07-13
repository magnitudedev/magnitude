import { Cause, Effect, Exit, Option, Schema } from 'effect'
import {
  makeAddressSpaceRuntime,
  type AddressSpaceRuntime,
  type AddressSpaceMutation,
  type AddressSpaceTransaction
} from '../addressed/address-space'
import { makeSchemaCodec } from '../addressed/codec'
import {
  childAddress,
  collectionSentinelAddress,
  isCollectionSentinelAddress,
  joinAddress
} from '../addressed/collections/address'
import {
  AddressedSequenceIndexSchema,
  makeAddressedSequence,
  type AddressedSequence,
  type AddressedSequenceIndex,
  type AddressedSequenceItem,
  type AddressedSequenceSegment,
  type AddressedSequenceWindowPart
} from '../addressed/collections/sequence'
import type { AddressedEntryStore } from '../addressed/entry-store'
import type { AddressedError } from '../addressed/errors'

/**
 * Callback into the projection bus so addressed content changes can trigger
 * consumer rebuilds. Called by `publish` for each changed property.
 */
export type AddressedChangeNotify = (
  sourceProjection: string,
  property: string,
  changedAddresses: ReadonlySet<string>
) => void

export interface ProjectionAddressedDescriptorBase<
  IndexSchema extends Schema.Schema.AnyNoContext,
  Entry,
  Handle,
  Consumer
> {
  readonly _tag: 'Sequence' | 'Record'
  readonly empty: Schema.Schema.Type<IndexSchema>
  readonly indexSchema: IndexSchema
  readonly makeRuntime: (
    namespace: string
  ) => Effect.Effect<AddressSpaceRuntime<Entry>, never, AddressedEntryStore>
  readonly makeHandle: (
    prefix: string,
    runtime: AddressSpaceRuntime<Entry>,
    transaction: AddressSpaceTransaction<Entry>
  ) => Handle
  readonly makeConsumer: (
    prefix: string,
    runtime: AddressSpaceRuntime<Entry>
  ) => Consumer
}

export interface ProjectionAddressedSequenceHandle<Item extends AddressedSequenceItem> {
  readonly empty: AddressedSequenceIndex
  readonly append: (
    index: AddressedSequenceIndex,
    item: Item
  ) => Effect.Effect<AddressedSequenceIndex, AddressedError>
  readonly updateById: (
    index: AddressedSequenceIndex,
    itemId: string,
    update: (item: Item) => Item
  ) => Effect.Effect<AddressedSequenceIndex, AddressedError>
  readonly removeById: (
    index: AddressedSequenceIndex,
    itemId: string
  ) => Effect.Effect<AddressedSequenceIndex, AddressedError>
  readonly replaceAll: (
    index: AddressedSequenceIndex,
    items: readonly Item[]
  ) => Effect.Effect<AddressedSequenceIndex, AddressedError>
  readonly replaceRange: (
    index: AddressedSequenceIndex,
    start: number,
    end: number,
    items: readonly Item[]
  ) => Effect.Effect<AddressedSequenceIndex, AddressedError>
  readonly positionOfItem: AddressedSequence<Item>['positionOfItem']
  readonly resolveAddressForItem: AddressedSequence<Item>['resolveAddressForItem']
  readonly resolveRangeWindow: AddressedSequence<Item>['resolveRangeWindow']
  readonly resolveTailWindow: AddressedSequence<Item>['resolveTailWindow']
  readonly readAll: (
    index: AddressedSequenceIndex
  ) => Effect.Effect<readonly Item[], AddressedError>
  readonly readWindow: (
    window: readonly AddressedSequenceWindowPart[]
  ) => Effect.Effect<readonly Item[], AddressedError>
}

export interface ProjectionAddressedSequenceConsumer<Item extends AddressedSequenceItem> {
  /** Sentinel address representing this instance's index structure. */
  readonly sentinelAddress: string
  readonly positionOfItem: AddressedSequence<Item>['positionOfItem']
  readonly resolveAddressForItem: AddressedSequence<Item>['resolveAddressForItem']
  readonly resolveRangeWindow: AddressedSequence<Item>['resolveRangeWindow']
  readonly resolveTailWindow: AddressedSequence<Item>['resolveTailWindow']
  readonly readWindow: (
    window: readonly AddressedSequenceWindowPart[]
  ) => Effect.Effect<readonly Item[], AddressedError>
  readonly readAll: (
    index: AddressedSequenceIndex
  ) => Effect.Effect<readonly Item[], AddressedError>
}

export interface ProjectionAddressedSequenceDescriptor<Item extends AddressedSequenceItem>
  extends ProjectionAddressedDescriptorBase<
    typeof AddressedSequenceIndexSchema,
    AddressedSequenceSegment<Item>,
    ProjectionAddressedSequenceHandle<Item>,
    ProjectionAddressedSequenceConsumer<Item>
  > {
  readonly _tag: 'Sequence'
  readonly empty: AddressedSequenceIndex
  readonly indexSchema: typeof AddressedSequenceIndexSchema
  readonly makeRuntime: (
    namespace: string
  ) => Effect.Effect<AddressSpaceRuntime<AddressedSequenceSegment<Item>>, never, AddressedEntryStore>
  readonly makeHandle: (
    prefix: string,
    runtime: AddressSpaceRuntime<AddressedSequenceSegment<Item>>,
    transaction: AddressSpaceTransaction<AddressedSequenceSegment<Item>>
  ) => ProjectionAddressedSequenceHandle<Item>
  readonly makeConsumer: (
    prefix: string,
    runtime: AddressSpaceRuntime<AddressedSequenceSegment<Item>>
  ) => ProjectionAddressedSequenceConsumer<Item>
}

export interface ProjectionAddressedRecordIndex<ChildIndex> {
  readonly members: readonly string[]
  readonly children: Readonly<Record<string, ChildIndex>>
}

export interface ProjectionAddressedRecordEncodedIndex<ChildEncodedIndex> {
  readonly members: readonly string[]
  readonly children: Readonly<Record<string, ChildEncodedIndex>>
}

export interface ProjectionAddressedRecordHandle<
  Member extends string,
  Child extends ProjectionAddressedDescriptor
> {
  readonly empty: ProjectionAddressedRecordIndex<ProjectionAddressedIndex<Child>>
  readonly has: (
    index: ProjectionAddressedRecordIndex<ProjectionAddressedIndex<Child>>,
    member: Member
  ) => boolean
  readonly members: (
    index: ProjectionAddressedRecordIndex<ProjectionAddressedIndex<Child>>
  ) => readonly string[]
  readonly resolveMember: (
    index: ProjectionAddressedRecordIndex<ProjectionAddressedIndex<Child>>,
    member: Member
  ) => Option.Option<ProjectionAddressedIndex<Child>>
  readonly ensure: (
    index: ProjectionAddressedRecordIndex<ProjectionAddressedIndex<Child>>,
    member: Member
  ) => ProjectionAddressedRecordIndex<ProjectionAddressedIndex<Child>>
  readonly remove: (
    index: ProjectionAddressedRecordIndex<ProjectionAddressedIndex<Child>>,
    member: Member
  ) => ProjectionAddressedRecordIndex<ProjectionAddressedIndex<Child>>
  readonly child: (member: Member) => ProjectionAddressedHandle<Child>
  readonly updateMember: (
    index: ProjectionAddressedRecordIndex<ProjectionAddressedIndex<Child>>,
    member: Member,
    update: (
      childIndex: ProjectionAddressedIndex<Child>,
      child: ProjectionAddressedHandle<Child>
    ) => Effect.Effect<ProjectionAddressedIndex<Child>, AddressedError>
  ) => Effect.Effect<ProjectionAddressedRecordIndex<ProjectionAddressedIndex<Child>>, AddressedError>
  readonly readMember: <A>(
    index: ProjectionAddressedRecordIndex<ProjectionAddressedIndex<Child>>,
    member: Member,
    read: (
      childIndex: ProjectionAddressedIndex<Child>,
      child: ProjectionAddressedHandle<Child>
    ) => Effect.Effect<A, AddressedError>
  ) => Effect.Effect<Option.Option<A>, AddressedError>
}

export interface ProjectionAddressedRecordConsumer<
  Member extends string,
  Child extends ProjectionAddressedDescriptor
> {
  /** Sentinel address representing this instance's membership structure. */
  readonly sentinelAddress: string
  readonly has: (
    index: ProjectionAddressedRecordIndex<ProjectionAddressedIndex<Child>>,
    member: Member
  ) => boolean
  readonly members: (
    index: ProjectionAddressedRecordIndex<ProjectionAddressedIndex<Child>>
  ) => readonly string[]
  readonly resolveMember: (
    index: ProjectionAddressedRecordIndex<ProjectionAddressedIndex<Child>>,
    member: Member
  ) => Option.Option<ProjectionAddressedIndex<Child>>
  readonly child: (member: Member) => ProjectionAddressedConsumer<Child>
  readonly readMember: <A>(
    index: ProjectionAddressedRecordIndex<ProjectionAddressedIndex<Child>>,
    member: Member,
    read: (
      childIndex: ProjectionAddressedIndex<Child>,
      child: ProjectionAddressedConsumer<Child>
    ) => Effect.Effect<A, AddressedError>
  ) => Effect.Effect<Option.Option<A>, AddressedError>
}

export interface ProjectionAddressedRecordDescriptor<
  Member extends string,
  Child extends ProjectionAddressedDescriptor
> extends ProjectionAddressedDescriptorBase<
  Schema.Schema<
    ProjectionAddressedRecordIndex<ProjectionAddressedIndex<Child>>,
    ProjectionAddressedRecordEncodedIndex<Schema.Schema.Encoded<Child['indexSchema']>>,
    never
  >,
  ProjectionAddressedEntry<Child>,
  ProjectionAddressedRecordHandle<Member, Child>,
  ProjectionAddressedRecordConsumer<Member, Child>
> {
  readonly _tag: 'Record'
  readonly empty: ProjectionAddressedRecordIndex<ProjectionAddressedIndex<Child>>
  readonly indexSchema: Schema.Schema<
    ProjectionAddressedRecordIndex<ProjectionAddressedIndex<Child>>,
    ProjectionAddressedRecordEncodedIndex<Schema.Schema.Encoded<Child['indexSchema']>>,
    never
  >
  readonly child: Child
  readonly makeRuntime: (
    namespace: string
  ) => Effect.Effect<AddressSpaceRuntime<ProjectionAddressedEntry<Child>>, never, AddressedEntryStore>
  readonly makeHandle: (
    prefix: string,
    runtime: AddressSpaceRuntime<ProjectionAddressedEntry<Child>>,
    transaction: AddressSpaceTransaction<ProjectionAddressedEntry<Child>>
  ) => ProjectionAddressedRecordHandle<Member, Child>
  readonly makeConsumer: (
    prefix: string,
    runtime: AddressSpaceRuntime<ProjectionAddressedEntry<Child>>
  ) => ProjectionAddressedRecordConsumer<Member, Child>
}

// TypeScript has no existential generic for "some addressed descriptor".
// The precise types stay attached to concrete descriptors through the associated
// Index/Entry/Handle/Consumer positions above; this alias is only for dynamic
// descriptor maps discovered at runtime.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type ProjectionAddressedDescriptor = ProjectionAddressedDescriptorBase<Schema.Schema.AnyNoContext, any, any, any>

export type ProjectionAddressedDescriptors = Readonly<Record<string, ProjectionAddressedDescriptor>>

export type ProjectionAddressedIndex<Descriptor> =
  Descriptor extends ProjectionAddressedDescriptorBase<infer IndexSchema, infer _Entry, infer _Handle, infer _Consumer>
    ? Schema.Schema.Type<IndexSchema>
    : never

export type ProjectionAddressedEntry<Descriptor> =
  Descriptor extends ProjectionAddressedDescriptorBase<infer _IndexSchema, infer Entry, infer _Handle, infer _Consumer>
    ? Entry
    : never

export type ProjectionAddressedHandle<Descriptor> =
  Descriptor extends ProjectionAddressedDescriptorBase<infer _IndexSchema, infer _Entry, infer Handle, infer _Consumer>
    ? Handle
    : never

export type ProjectionAddressedConsumer<Descriptor> =
  Descriptor extends ProjectionAddressedDescriptorBase<infer _IndexSchema, infer _Entry, infer _Handle, infer Consumer>
    ? Consumer
    : never

export type ProjectionAddressedHandles<TAddressed extends ProjectionAddressedDescriptors> = {
  readonly [K in keyof TAddressed]:
    TAddressed[K] extends ProjectionAddressedDescriptor
      ? ProjectionAddressedHandle<TAddressed[K]>
      : never
}

export type ProjectionAddressedConsumers<TAddressed extends ProjectionAddressedDescriptors> = {
  readonly [K in keyof TAddressed]:
    TAddressed[K] extends ProjectionAddressedDescriptor
      ? ProjectionAddressedConsumer<TAddressed[K]>
      : never
}

// ---------------------------------------------------------------------------
// Consumer view types — how addressed fields appear through read()
// ---------------------------------------------------------------------------

/** How one addressed descriptor's field appears to a consuming projection. */
export type ProjectionAddressedView<Descriptor> =
  Descriptor extends ProjectionAddressedSequenceDescriptor<infer Item>
    ? readonly Item[]
    : Descriptor extends ProjectionAddressedRecordDescriptor<string, infer Child>
      ? { readonly [member: string]: ProjectionAddressedView<Child> | undefined }
      : never

/**
 * A projection's state as seen by consumers through read(): addressed index
 * fields are replaced by their native-type views. The runtime Proxies make
 * this true at runtime; this type makes it true statically — no casts at the
 * read seam.
 */
export type AddressedConsumerState<TState, TAddressed extends ProjectionAddressedDescriptors> =
  keyof TAddressed extends never
    ? TState
    : {
        readonly [K in keyof TState]: K extends keyof TAddressed
          ? ProjectionAddressedView<TAddressed[K]>
          : TState[K]
      }

export interface ProjectionForkedAddressedHandles<TAddressed extends ProjectionAddressedDescriptors> {
  readonly forFork: (forkId: string | null) => ProjectionAddressedHandles<TAddressed>
}

export type ProjectionForkedAddressedConsumers<TAddressed extends ProjectionAddressedDescriptors> =
  ProjectionAddressedConsumers<TAddressed> & {
    readonly forFork: (forkId: string | null) => ProjectionAddressedConsumers<TAddressed>
  }

export interface ProjectionAddressedMutation<TAddressed extends ProjectionAddressedDescriptors> {
  readonly handles: ProjectionAddressedHandles<TAddressed>
  readonly handlesFor: (scope: Iterable<string>) => ProjectionAddressedHandles<TAddressed>
}

export interface ProjectionAddressedTransactionResult<A> {
  readonly value: A
  readonly changed: ReadonlyMap<string, ReadonlySet<string>>
}

export interface ProjectionAddressedRuntime<TAddressed extends ProjectionAddressedDescriptors> {
  readonly isEmpty: boolean
  readonly descriptors: TAddressed
  readonly transact: <A, E, R>(
    body: (mutation: ProjectionAddressedMutation<TAddressed>) => Effect.Effect<A, E, R>
  ) => Effect.Effect<ProjectionAddressedTransactionResult<A>, E | AddressedError, R>
  readonly publish: (changed: ReadonlyMap<string, ReadonlySet<string>>) => Effect.Effect<void>
  /**
   * Replace a consumer's pins for one property's address space. Sentinel
   * addresses in the set are notification-only and are filtered out here.
   */
  readonly pinConsumer: (
    property: string,
    owner: string,
    addresses: ReadonlySet<string>
  ) => Effect.Effect<void, AddressedError>
  readonly consumers: ProjectionAddressedConsumers<TAddressed>
  readonly consumersFor: (scope: Iterable<string>) => ProjectionAddressedConsumers<TAddressed>
  readonly flushDirty: Effect.Effect<void, AddressedError>
  readonly reset: Effect.Effect<void>
}

// ---------------------------------------------------------------------------
// Read tracker — handler-scoped, keyed by (source projection, property)
// ---------------------------------------------------------------------------

/**
 * Addresses read through consumer proxies during one handler invocation,
 * keyed by source projection name, then by addressed property. Includes
 * collection sentinels (notification-only; filtered out when pinning).
 * Created at handler start; no module-level state.
 */
export type AddressedReadTracker = Map<string, Map<string, Set<string>>>

export const makeReadTracker = (): AddressedReadTracker => new Map()

export const trackRead = (
  tracker: AddressedReadTracker,
  source: string,
  property: string,
  address: string
): void => {
  let perProperty = tracker.get(source)
  if (!perProperty) {
    perProperty = new Map()
    tracker.set(source, perProperty)
  }
  let addresses = perProperty.get(property)
  if (!addresses) {
    addresses = new Set()
    perProperty.set(property, addresses)
  }
  addresses.add(address)
}

// ---------------------------------------------------------------------------
// Proxy: addressed sequence as readonly array
// ---------------------------------------------------------------------------

/**
 * Run a read effect synchronously inside a Proxy trap. Resident segments are
 * pure memory; non-resident segments load from the store as part of the read
 * (occasional I/O, once per segment — the loaded value is retained resident
 * and pinned by the framework right after the handler).
 */
const runSync = <A, E>(effect: Effect.Effect<A, E>): A => {
  const exit = Effect.runSyncExit(effect)
  if (Exit.isSuccess(exit)) return exit.value
  throw Cause.squash(exit.cause)
}

/**
 * Whole-array methods served by materializing the sequence once and binding
 * the real array's own method — native semantics for every overload.
 * `at`/`slice`/index access have window-optimized paths instead.
 */
const wholeArrayMethodNames = [
  'map', 'forEach', 'filter', 'find', 'findIndex', 'findLast', 'findLastIndex',
  'some', 'every', 'reduce', 'reduceRight', 'flatMap', 'join', 'includes',
  'indexOf', 'lastIndexOf', 'entries', 'keys', 'values',
  'toString', 'toLocaleString'
] as const
type WholeArrayMethod = (typeof wholeArrayMethodNames)[number]
const wholeArrayMethodSet: ReadonlySet<string> = new Set(wholeArrayMethodNames)
const isWholeArrayMethod = (prop: string): prop is WholeArrayMethod =>
  wholeArrayMethodSet.has(prop)

/**
 * Create a Proxy that presents an addressed sequence as a readonly array.
 *
 * - `length` → resolved from the index (no I/O)
 * - Numeric index / `slice` / iteration / whitelisted array methods → segment
 *   reads via `runSync` (resident = memory, non-resident = synchronous load)
 * - Every access records the instance sentinel; reads also record the touched
 *   segment addresses
 * - Non-whitelisted string properties resolve to `undefined` — no catch-all
 *   that reads the full history on property probes (`then`, `constructor`, …)
 */
export const makeSequenceProxy = <Item extends AddressedSequenceItem>(
  index: AddressedSequenceIndex,
  consumer: ProjectionAddressedSequenceConsumer<Item>,
  record: (address: string) => void
): readonly Item[] => {
  const touch = () => record(consumer.sentinelAddress)

  const readRange = (start: number, limit: number): readonly Item[] => {
    const window = consumer.resolveRangeWindow(index, start, limit)
    for (const part of window) record(part.address)
    return runSync(consumer.readWindow(window))
  }

  const readAll = (): readonly Item[] => readRange(0, index.totalCount)

  const readAt = (position: number): Item | undefined => {
    if (position < 0 || position >= index.totalCount) return undefined
    return readRange(position, 1)[0]
  }

  const at = (i: number): Item | undefined =>
    readAt(i < 0 ? index.totalCount + i : i)

  const slice = (start?: number, end?: number): readonly Item[] => {
    const len = index.totalCount
    const s = start === undefined ? 0 : start < 0 ? Math.max(0, len + start) : Math.min(start, len)
    const e = end === undefined ? len : end < 0 ? Math.max(0, len + end) : Math.min(end, len)
    if (s >= e) return []
    return readRange(s, e - s)
  }

  const handler: ProxyHandler<Item[]> = {
    get(target, prop, receiver) {
      touch()
      if (prop === 'length') return index.totalCount
      if (prop === Symbol.iterator) {
        return function* (): Iterator<Item> {
          yield* readAll()
        }
      }
      if (typeof prop === 'string') {
        if (/^\d+$/.test(prop)) return readAt(Number(prop))
        if (prop === 'at') return at
        if (prop === 'slice') return slice
        if (isWholeArrayMethod(prop)) {
          // Delegate to the real array's own method — native semantics for
          // every overload, no reimplementation.
          const items = readAll()
          return items[prop].bind(items)
        }
        return undefined
      }
      return Reflect.get(target, prop, receiver)
    },
    has(_target, prop) {
      touch()
      if (prop === 'length' || prop === Symbol.iterator) return true
      if (typeof prop === 'string') {
        if (/^\d+$/.test(prop)) return Number(prop) < index.totalCount
        return prop === 'at' || prop === 'slice' || isWholeArrayMethod(prop)
      }
      return false
    },
    ownKeys() {
      touch()
      return ['length', ...Array.from({ length: index.totalCount }, (_, i) => String(i))]
    },
    getOwnPropertyDescriptor(_target, prop) {
      touch()
      if (prop === 'length') {
        // writable: true satisfies the proxy invariants against the `[]`
        // target, whose own `length` is writable and non-configurable.
        return { value: index.totalCount, writable: true, enumerable: false, configurable: false }
      }
      if (typeof prop === 'string' && /^\d+$/.test(prop)) {
        const i = Number(prop)
        if (i >= 0 && i < index.totalCount) {
          return { value: readAt(i), writable: false, enumerable: true, configurable: true }
        }
      }
      return undefined
    }
  }

  const target: Item[] = []
  return new Proxy(target, handler)
}

// ---------------------------------------------------------------------------
// Proxy: addressed record as readonly object
// ---------------------------------------------------------------------------

/**
 * Create a Proxy that presents an addressed record as a readonly plain
 * object: member access, `in`, `Object.keys`/`entries`, and spread all work.
 * Every access records the record's sentinel (membership structure); member
 * values are child proxies that do their own recording.
 */
export const makeRecordProxy = (
  index: ProjectionAddressedRecordIndex<unknown>,
  sentinelAddress: string,
  record: (address: string) => void,
  makeChildProxy: (member: string, childIndex: unknown) => unknown
): { readonly [key: string]: unknown } => {
  const touch = () => record(sentinelAddress)

  const handler: ProxyHandler<{ readonly [key: string]: unknown }> = {
    get(target, prop, receiver) {
      touch()
      if (typeof prop === 'string') {
        const childIndex = index.children[prop]
        return childIndex === undefined ? undefined : makeChildProxy(prop, childIndex)
      }
      return Reflect.get(target, prop, receiver)
    },
    has(_target, prop) {
      touch()
      return typeof prop === 'string' && prop in index.children
    },
    ownKeys() {
      touch()
      return Object.keys(index.children)
    },
    getOwnPropertyDescriptor(_target, prop) {
      touch()
      if (typeof prop === 'string' && prop in index.children) {
        return { value: makeChildProxy(prop, index.children[prop]), writable: false, enumerable: true, configurable: true }
      }
      return undefined
    }
  }

  return new Proxy({}, handler)
}

// ---------------------------------------------------------------------------
// State wrapping — replace addressed index fields with Proxies on read()
// ---------------------------------------------------------------------------

/**
 * Given a projection's committed state, replace each addressed field with a
 * Proxy that acts like the native type. `record(property, address)` receives
 * every touched address (sentinels and segments), keyed by the top-level
 * addressed property.
 *
 * This is called by the read() function when a consuming projection reads a
 * dependency's state. The descriptors and consumers come from the owning
 * projection's runtime.
 */
export const wrapStateWithProxies = <TState>(
  state: TState,
  descriptors: ProjectionAddressedDescriptors,
  consumers: ProjectionAddressedConsumers<ProjectionAddressedDescriptors>,
  record: (property: string, address: string) => void
): TState => {
  if (typeof state !== 'object' || state === null) return state

  const addressedKeys = Object.keys(descriptors)
  if (addressedKeys.length === 0) return state

  const wrapped: Record<string, unknown> = {}
  for (const key of addressedKeys) {
    const descriptor = descriptors[key]
    const consumer = (consumers as Record<string, unknown>)[key]
    const index = (state as Record<string, unknown>)[key]
    if (index === undefined || consumer === undefined) continue
    wrapped[key] = buildProxy(descriptor, index, consumer, (address) => record(key, address))
  }

  return new Proxy(state as Record<string, unknown>, {
    get(target, prop, receiver) {
      if (typeof prop === 'string' && prop in wrapped) {
        return wrapped[prop]
      }
      return Reflect.get(target, prop, receiver)
    },
    has(target, prop) {
      if (typeof prop === 'string' && prop in wrapped) return true
      return Reflect.has(target, prop)
    },
    ownKeys(target) {
      return Reflect.ownKeys(target)
    },
    getOwnPropertyDescriptor(target, prop) {
      if (typeof prop === 'string' && prop in wrapped) {
        return { value: wrapped[prop], writable: false, enumerable: true, configurable: true }
      }
      return Reflect.getOwnPropertyDescriptor(target, prop)
    }
  }) as TState
}

// ---------------------------------------------------------------------------
// Tracking consumers — direct consumer operations with the same read semantics
// ---------------------------------------------------------------------------

export const wrapConsumersWithTracking = <TAddressed extends ProjectionAddressedDescriptors>(
  descriptors: TAddressed,
  consumers: ProjectionAddressedConsumers<TAddressed>,
  record: (property: string, address: string) => void
): ProjectionAddressedConsumers<TAddressed> => {
  const wrapped: Record<string, unknown> = {}

  for (const [property, descriptor] of Object.entries(descriptors)) {
    const consumer = (consumers as Record<string, unknown>)[property]
    if (consumer === undefined) continue
    wrapped[property] = wrapConsumerWithTracking(
      descriptor,
      consumer,
      (address) => record(property, address)
    )
  }

  return wrapped as ProjectionAddressedConsumers<TAddressed>
}

/**
 * Build the Proxy for one descriptor's index: a sequence array Proxy, or a
 * record Proxy whose members recurse through this same builder.
 */
const buildProxy = (
  descriptor: ProjectionAddressedDescriptor,
  index: unknown,
  consumer: unknown,
  record: (address: string) => void
): unknown => {
  if (descriptor._tag === 'Sequence') {
    return makeSequenceProxy(
      index as AddressedSequenceIndex,
      consumer as ProjectionAddressedSequenceConsumer<AddressedSequenceItem>,
      record
    )
  }
  const recDescriptor = descriptor as ProjectionAddressedRecordDescriptor<string, ProjectionAddressedDescriptor>
  const recConsumer = consumer as ProjectionAddressedRecordConsumer<string, ProjectionAddressedDescriptor>
  return makeRecordProxy(
    index as ProjectionAddressedRecordIndex<unknown>,
    recConsumer.sentinelAddress,
    record,
    (member, childIndex) => buildProxy(recDescriptor.child, childIndex, recConsumer.child(member), record)
  )
}

const wrapConsumerWithTracking = (
  descriptor: ProjectionAddressedDescriptor,
  consumer: unknown,
  record: (address: string) => void
): unknown => {
  if (descriptor._tag === 'Sequence') {
    const seq = consumer as ProjectionAddressedSequenceConsumer<AddressedSequenceItem>
    const touch = () => record(seq.sentinelAddress)

    return {
      sentinelAddress: seq.sentinelAddress,
      positionOfItem: (index, itemId) => {
        touch()
        return seq.positionOfItem(index, itemId)
      },
      resolveAddressForItem: (index, itemId) => {
        touch()
        return seq.resolveAddressForItem(index, itemId)
      },
      resolveRangeWindow: (index, start, limit) => {
        touch()
        return seq.resolveRangeWindow(index, start, limit)
      },
      resolveTailWindow: (index, limit) => {
        touch()
        return seq.resolveTailWindow(index, limit)
      },
      readWindow: (window) => {
        for (const part of window) record(part.address)
        return seq.readWindow(window)
      },
      readAll: (index) => {
        touch()
        for (const segment of index.segments) record(segment.address)
        return seq.readAll(index)
      }
    } satisfies ProjectionAddressedSequenceConsumer<AddressedSequenceItem>
  }

  const recDescriptor = descriptor as ProjectionAddressedRecordDescriptor<string, ProjectionAddressedDescriptor>
  const rec = consumer as ProjectionAddressedRecordConsumer<string, ProjectionAddressedDescriptor>
  const touch = () => record(rec.sentinelAddress)

  return {
    sentinelAddress: rec.sentinelAddress,
    has: (index, member) => {
      touch()
      return rec.has(index, member)
    },
    members: (index) => {
      touch()
      return rec.members(index)
    },
    resolveMember: (index, member) => {
      touch()
      return rec.resolveMember(index, member)
    },
    child: (member) => {
      touch()
      return wrapConsumerWithTracking(recDescriptor.child, rec.child(member), record)
    },
    readMember: (index, member, read) => {
      touch()
      return rec.readMember(index, member, (childIndex, child) =>
        read(
          childIndex,
          wrapConsumerWithTracking(recDescriptor.child, child, record) as ProjectionAddressedConsumer<ProjectionAddressedDescriptor>
        )
      )
    }
  } satisfies ProjectionAddressedRecordConsumer<string, ProjectionAddressedDescriptor>
}

export const sequence = <Item extends AddressedSequenceItem, Encoded>(
  itemSchema: Schema.Schema<Item, Encoded, never>
): ProjectionAddressedSequenceDescriptor<Item> => {
  const segmentSchema = Schema.Struct({
    items: Schema.Array(itemSchema)
  })

  return {
    _tag: 'Sequence',
    empty: {
      nextSegmentNumber: 0,
      nextAddressNumber: 0,
      totalCount: 0,
      segments: []
    },
    indexSchema: AddressedSequenceIndexSchema,
    makeRuntime: (namespace) =>
      makeAddressSpaceRuntime<AddressedSequenceSegment<Item>>({
        namespace,
        codec: makeSchemaCodec(segmentSchema)
      }),
    makeHandle: (prefix, runtime, transaction) => {
      const collection = makeAddressedSequence({
        prefix,
        runtime
      })

      // Structural changes (any operation that returns a new index) mark the
      // instance sentinel, so consumers tracking this sequence are notified
      // even when the change allocated a fresh segment they never read.
      // Content-only updates return the input index by reference and don't.
      const marked = (
        index: AddressedSequenceIndex,
        operation: Effect.Effect<AddressedSequenceIndex, AddressedError>
      ): Effect.Effect<AddressedSequenceIndex, AddressedError> =>
        Effect.tap(operation, (next) =>
          Effect.sync(() => {
            if (next !== index) transaction.markChanged(collection.sentinelAddress)
          })
        )

      return {
        empty: collection.empty,
        append: (index, item) => marked(index, collection.append(transaction, index, item)),
        updateById: (index, itemId, update) =>
          marked(index, collection.updateById(transaction, index, itemId, update)),
        removeById: (index, itemId) => marked(index, collection.removeById(transaction, index, itemId)),
        replaceAll: (index, items) => marked(index, collection.replaceAll(transaction, index, items)),
        replaceRange: (index, start, end, items) =>
          marked(index, collection.replaceRange(transaction, index, start, end, items)),
        positionOfItem: collection.positionOfItem,
        resolveAddressForItem: collection.resolveAddressForItem,
        resolveRangeWindow: collection.resolveRangeWindow,
        resolveTailWindow: collection.resolveTailWindow,
        readAll: (index) => collection.readAllInTransaction(transaction, index),
        readWindow: (window) => collection.readWindowInTransaction(transaction, window)
      }
    },
    makeConsumer: (prefix, runtime) => {
      const collection = makeAddressedSequence({
        prefix,
        runtime
      })

      return {
        sentinelAddress: collection.sentinelAddress,
        positionOfItem: collection.positionOfItem,
        resolveAddressForItem: collection.resolveAddressForItem,
        resolveRangeWindow: collection.resolveRangeWindow,
        resolveTailWindow: collection.resolveTailWindow,
        readWindow: collection.readWindow,
        readAll: collection.readAll
      }
    }
  }
}

const makeRecordIndexSchema = <Child extends ProjectionAddressedDescriptor>(
  child: Child
): Schema.Schema<
  ProjectionAddressedRecordIndex<ProjectionAddressedIndex<Child>>,
  ProjectionAddressedRecordEncodedIndex<Schema.Schema.Encoded<Child['indexSchema']>>,
  never
> =>
  Schema.Struct({
    members: Schema.Array(Schema.String),
    children: Schema.Record({
      key: Schema.String,
      value: child.indexSchema
    })
  }).pipe(
    Schema.filter(
      (index) => {
        const members = new Set(index.members)
        if (members.size !== index.members.length) return false

        const childKeys = Object.keys(index.children)
        if (childKeys.length !== members.size) return false

        for (const member of members) {
          if (!Object.prototype.hasOwnProperty.call(index.children, member)) return false
        }
        return true
      },
      {
        message: () => 'record members must be unique and exactly match child index keys'
      }
    )
  )

const emptyRecordIndex = <Child extends ProjectionAddressedDescriptor>(
  child: Child
): ProjectionAddressedRecordIndex<ProjectionAddressedIndex<Child>> => ({
  members: [],
  children: {}
})

const hasRecordMember = <Child extends ProjectionAddressedDescriptor>(
  index: ProjectionAddressedRecordIndex<ProjectionAddressedIndex<Child>>,
  member: string
): boolean =>
  Object.prototype.hasOwnProperty.call(index.children, member)

const recordMembers = <Child extends ProjectionAddressedDescriptor>(
  index: ProjectionAddressedRecordIndex<ProjectionAddressedIndex<Child>>
): readonly string[] =>
  index.members

const resolveRecordMember = <Child extends ProjectionAddressedDescriptor>(
  index: ProjectionAddressedRecordIndex<ProjectionAddressedIndex<Child>>,
  member: string
): Option.Option<ProjectionAddressedIndex<Child>> => {
  if (!hasRecordMember(index, member)) return Option.none()

  const childIndex = index.children[member]
  return childIndex === undefined
    ? Option.none()
    : Option.some(childIndex)
}

const ensureRecordMember = <Child extends ProjectionAddressedDescriptor>(
  child: Child,
  index: ProjectionAddressedRecordIndex<ProjectionAddressedIndex<Child>>,
  member: string
): ProjectionAddressedRecordIndex<ProjectionAddressedIndex<Child>> =>
  Object.prototype.hasOwnProperty.call(index.children, member)
    ? index
    : {
        members: index.members.includes(member) ? index.members : [...index.members, member],
        children: {
          ...index.children,
          [member]: child.empty
        }
      }

const removeRecordMember = <Child extends ProjectionAddressedDescriptor>(
  index: ProjectionAddressedRecordIndex<ProjectionAddressedIndex<Child>>,
  member: string
): ProjectionAddressedRecordIndex<ProjectionAddressedIndex<Child>> => {
  if (!hasRecordMember(index, member)) return index

  const children: Record<string, ProjectionAddressedIndex<Child>> = {}
  for (const [childMember, childIndex] of Object.entries(index.children)) {
    if (childMember !== member) {
      children[childMember] = childIndex
    }
  }

  return {
    members: index.members.filter((candidate) => candidate !== member),
    children
  }
}

export const record = <
  Child extends ProjectionAddressedDescriptor,
  Member extends string = string
>(
  child: Child
): ProjectionAddressedRecordDescriptor<Member, Child> => ({
  _tag: 'Record',
  empty: emptyRecordIndex(child),
  indexSchema: makeRecordIndexSchema(child),
  child,
  makeRuntime: (namespace) =>
    child.makeRuntime(namespace),
  makeHandle: (prefix, runtime, transaction) => {
    const sentinelAddress = collectionSentinelAddress(prefix)
    const childFor = (member: Member): ProjectionAddressedHandle<Child> =>
      child.makeHandle(
        childAddress(prefix, 'members', member),
        runtime,
        transaction
      )

    // Membership changes mark the record's sentinel so consumers tracking
    // this record's structure (keys, `in`) are notified. Child content and
    // structure changes are marked by the child handles themselves.
    const markedMembership = (
      index: ProjectionAddressedRecordIndex<ProjectionAddressedIndex<Child>>,
      next: ProjectionAddressedRecordIndex<ProjectionAddressedIndex<Child>>
    ): ProjectionAddressedRecordIndex<ProjectionAddressedIndex<Child>> => {
      if (next !== index) transaction.markChanged(sentinelAddress)
      return next
    }

    return {
      empty: emptyRecordIndex(child),
      has: hasRecordMember,
      members: recordMembers,
      resolveMember: resolveRecordMember,
      ensure: (index, member) => markedMembership(index, ensureRecordMember(child, index, member)),
      remove: (index, member) => markedMembership(index, removeRecordMember(index, member)),
      child: childFor,
      updateMember: (index, member, update) =>
        Effect.gen(function* () {
          const hadMember = hasRecordMember(index, member)
          const nextIndex = ensureRecordMember(child, index, member)
          if (!hadMember) transaction.markChanged(sentinelAddress)
          const childIndex = nextIndex.children[member]
          const updatedChild = yield* update(childIndex, childFor(member))
          if (hadMember && updatedChild === childIndex) {
            return index
          }
          return {
            ...nextIndex,
            children: {
              ...nextIndex.children,
              [member]: updatedChild
            }
          }
        }),
      readMember: (index, member, read) => {
        const childIndex = index.children[member]
        return childIndex === undefined
          ? Effect.succeed(Option.none())
          : Effect.map(read(childIndex, childFor(member)), Option.some)
      }
    }
  },
  makeConsumer: (prefix, runtime) => {
    const childFor = (member: Member): ProjectionAddressedConsumer<Child> =>
      child.makeConsumer(
        childAddress(prefix, 'members', member),
        runtime
      )

    return {
      sentinelAddress: collectionSentinelAddress(prefix),
      has: hasRecordMember,
      members: recordMembers,
      resolveMember: resolveRecordMember,
      child: childFor,
      readMember: (index, member, read) => {
        const childIndex = index.children[member]
        return childIndex === undefined
          ? Effect.succeed(Option.none())
          : Effect.map(read(childIndex, childFor(member)), Option.some)
      }
    }
  }
})

const emptyProjectionAddressedRuntime = <TAddressed extends ProjectionAddressedDescriptors>(): ProjectionAddressedRuntime<TAddressed> => ({
  isEmpty: true,
  descriptors: {} as TAddressed,
  transact: (body) =>
    Effect.map(
      body({
        handles: {} as ProjectionAddressedHandles<TAddressed>,
        handlesFor: () => ({} as ProjectionAddressedHandles<TAddressed>)
      }),
      (value) => ({
        value,
        changed: new Map()
      })
    ),
  publish: () => Effect.void,
  pinConsumer: () => Effect.void,
  consumers: {} as ProjectionAddressedConsumers<TAddressed>,
  consumersFor: () => ({} as ProjectionAddressedConsumers<TAddressed>),
  flushDirty: Effect.void,
  reset: Effect.void
})

interface ProjectionAddressedRuntimeSlotMutation {
  readonly makeHandle: (prefix: string) => unknown
  readonly commit: Effect.Effect<ReadonlySet<string>, AddressedError>
}

interface ProjectionAddressedRuntimeSlot {
  readonly makeMutation: () => ProjectionAddressedRuntimeSlotMutation
  readonly makeConsumer: (prefix: string) => unknown
  readonly pin: (owner: string, addresses: Iterable<string>) => Effect.Effect<void, AddressedError>
  readonly flushDirty: Effect.Effect<void, AddressedError>
  readonly reset: Effect.Effect<void>
}

const makeProjectionAddressedRuntimeSlot = <Descriptor extends ProjectionAddressedDescriptor>(
  descriptor: Descriptor,
  runtime: AddressSpaceRuntime<ProjectionAddressedEntry<Descriptor>>
): ProjectionAddressedRuntimeSlot => ({
  makeMutation: () => {
    const mutation: AddressSpaceMutation<ProjectionAddressedEntry<Descriptor>> = runtime.makeMutation()
    return {
      makeHandle: (prefix) => descriptor.makeHandle(prefix, runtime, mutation.transaction),
      commit: mutation.commit
    }
  },
  makeConsumer: (prefix) => descriptor.makeConsumer(prefix, runtime),
  pin: runtime.pin,
  flushDirty: runtime.flushDirty,
  reset: runtime.reset
})

export const makeProjectionAddressedRuntime = <TAddressed extends ProjectionAddressedDescriptors>(
  projectionName: string,
  descriptors: TAddressed,
  notify: AddressedChangeNotify
): Effect.Effect<ProjectionAddressedRuntime<TAddressed>, never, AddressedEntryStore> =>
  Effect.gen(function* () {
    const entries: Array<readonly [string, ProjectionAddressedDescriptor]> = Object.entries(descriptors)
    if (entries.length === 0) {
      return emptyProjectionAddressedRuntime<TAddressed>()
    }

    const slots = new Map<string, ProjectionAddressedRuntimeSlot>()
    for (const [property, descriptor] of entries) {
      const namespace = `${projectionName}/${property}`
      const runtime = yield* descriptor.makeRuntime(namespace)
      slots.set(property, makeProjectionAddressedRuntimeSlot(descriptor, runtime))
    }

    const consumersFor = (scope: Iterable<string>): ProjectionAddressedConsumers<TAddressed> => {
      const scopeParts = [...scope]
      const consumers: Record<string, unknown> = {}

      for (const [property] of entries) {
        const slot = slots.get(property)
        if (!slot) continue
        consumers[property] = slot.makeConsumer(joinAddress([projectionName, property, ...scopeParts]))
      }

      return consumers as ProjectionAddressedConsumers<TAddressed>
    }

    const transact: ProjectionAddressedRuntime<TAddressed>['transact'] = (body) => {
      const mutations = new Map<string, ProjectionAddressedRuntimeSlotMutation>()

      for (const [property] of entries) {
        const slot = slots.get(property)
        if (!slot) continue
        mutations.set(property, slot.makeMutation())
      }

      const handlesFor = (scope: Iterable<string>): ProjectionAddressedHandles<TAddressed> => {
        const scopeParts = [...scope]
        const handles: Record<string, unknown> = {}

        for (const [property] of entries) {
          const mutation = mutations.get(property)
          if (!mutation) continue
          handles[property] = mutation.makeHandle(joinAddress([projectionName, property, ...scopeParts]))
        }

        return handles as ProjectionAddressedHandles<TAddressed>
      }

      return Effect.gen(function* () {
        const value = yield* body({
          handles: handlesFor([]),
          handlesFor
        })

        const changed = new Map<string, ReadonlySet<string>>()
        for (const [property, mutation] of mutations) {
          const addresses = yield* mutation.commit
          if (addresses.size > 0) changed.set(property, addresses)
        }

        return { value, changed }
      })
    }

    return {
      isEmpty: false,
      descriptors,
      transact,
      publish: (changed) =>
        Effect.sync(() => {
          for (const [property, addresses] of changed) {
            notify(projectionName, property, addresses)
          }
        }),
      pinConsumer: (property, owner, addresses) => {
        const slot = slots.get(property)
        if (!slot) {
          return Effect.dieMessage(
            `Projection "${projectionName}" has no addressed property "${property}" to pin`
          )
        }
        const pinnable = [...addresses].filter((address) => !isCollectionSentinelAddress(address))
        return slot.pin(owner, pinnable)
      },
      consumers: consumersFor([]),
      consumersFor,

      flushDirty: Effect.forEach(
        slots.values(),
        (slot) => slot.flushDirty,
        { discard: true }
      ),

      reset: Effect.forEach(
        slots.values(),
        (slot) => slot.reset,
        { discard: true }
      )
    }
  })
