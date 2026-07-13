import { Effect, Option, Schema } from 'effect'
import type { AddressSpaceRuntime, AddressSpaceTransaction } from '../address-space'
import { AddressedCollectionError, type AddressedError } from '../errors'
import { childAddress, collectionSentinelAddress } from './address'

export const ADDRESSED_SEQUENCE_SEGMENT_CAPACITY = 50

export interface AddressedSequenceItem {
  readonly id: string
}

export interface AddressedSequenceSegment<Item extends AddressedSequenceItem> {
  readonly items: readonly Item[]
}

export interface AddressedSequenceSegmentIndex {
  readonly id: string
  readonly address: string
  readonly start: number
  readonly count: number
  readonly itemIds: readonly string[]
}

export interface AddressedSequenceIndex {
  readonly nextSegmentNumber: number
  readonly nextAddressNumber: number
  readonly totalCount: number
  readonly segments: readonly AddressedSequenceSegmentIndex[]
}

const NonNegativeInt = Schema.Number.pipe(Schema.int(), Schema.nonNegative())
const SegmentCount = Schema.Number.pipe(Schema.int(), Schema.between(0, ADDRESSED_SEQUENCE_SEGMENT_CAPACITY))

interface SequenceIndexIssue {
  readonly segment?: AddressedSequenceSegmentIndex
  readonly reason: string
}

const entryId = (number: number): string =>
  `entry-${number}`

const allocatedEntryNumber = (address: string): number | undefined => {
  const match = /(?:^|\/)entries\/entry-(\d+)$/.exec(address)
  if (!match) return undefined
  const value = Number(match[1])
  return Number.isSafeInteger(value) ? value : undefined
}

const sequenceIndexIssue = (
  index: AddressedSequenceIndex
): SequenceIndexIssue | undefined => {
  const nextAddressNumber = index.nextAddressNumber

  if (!Number.isInteger(index.nextSegmentNumber) || index.nextSegmentNumber < 0) {
    return {
      reason: `next segment number must be a non-negative integer; got ${index.nextSegmentNumber}`
    }
  }

  if (!Number.isInteger(nextAddressNumber) || nextAddressNumber < 0) {
    return {
      reason: `next address number must be a non-negative integer; got ${nextAddressNumber}`
    }
  }

  if (!Number.isInteger(index.totalCount) || index.totalCount < 0) {
    return {
      reason: `total count must be a non-negative integer; got ${index.totalCount}`
    }
  }

  if (index.nextSegmentNumber !== index.segments.length) {
    return {
      reason: `next segment number ${index.nextSegmentNumber} did not match ${index.segments.length} segment(s)`
    }
  }

  let expectedStart = 0
  const itemIds = new Set<string>()
  const addresses = new Set<string>()
  let maxAllocatedAddressNumber = -1
  for (const segment of index.segments) {
    if (!Number.isInteger(segment.start) || segment.start < 0) {
      return {
        segment,
        reason: `segment start must be a non-negative integer; got ${segment.start}`
      }
    }

    if (!Number.isInteger(segment.count) || segment.count < 0) {
      return {
        segment,
        reason: `segment count must be a non-negative integer; got ${segment.count}`
      }
    }

    if (segment.count > ADDRESSED_SEQUENCE_SEGMENT_CAPACITY) {
      return {
        segment,
        reason: `segment count ${segment.count} exceeds segment capacity ${ADDRESSED_SEQUENCE_SEGMENT_CAPACITY}`
      }
    }

    if (segment.itemIds.length !== segment.count) {
      return {
        segment,
        reason: `segment count ${segment.count} did not match ${segment.itemIds.length} item id(s)`
      }
    }

    if (addresses.has(segment.address)) {
      return {
        segment,
        reason: `segment address "${segment.address}" appears more than once in sequence index`
      }
    }
    addresses.add(segment.address)

    const allocatedAddressNumber = allocatedEntryNumber(segment.address)
    if (allocatedAddressNumber !== undefined) {
      maxAllocatedAddressNumber = Math.max(maxAllocatedAddressNumber, allocatedAddressNumber)
    }

    if (segment.start !== expectedStart) {
      return {
        segment,
        reason: `segment start ${segment.start} did not match expected start ${expectedStart}`
      }
    }

    for (const itemId of segment.itemIds) {
      if (itemIds.has(itemId)) {
        return {
          segment,
          reason: `item id "${itemId}" appears more than once in sequence index`
        }
      }
      itemIds.add(itemId)
    }

    expectedStart += segment.count
  }

  if (nextAddressNumber <= maxAllocatedAddressNumber) {
    return {
      reason: `next address number ${nextAddressNumber} did not exceed allocated entry ${maxAllocatedAddressNumber}`
    }
  }

  return expectedStart === index.totalCount
    ? undefined
    : {
        reason: `total count ${index.totalCount} did not match ${expectedStart} indexed item(s)`
      }
}

export const AddressedSequenceSegmentIndexSchema = Schema.Struct({
  id: Schema.String,
  address: Schema.String,
  start: NonNegativeInt,
  count: SegmentCount,
  itemIds: Schema.Array(Schema.String)
})

export const AddressedSequenceIndexSchema = Schema.Struct({
  nextSegmentNumber: NonNegativeInt,
  nextAddressNumber: NonNegativeInt,
  totalCount: NonNegativeInt,
  segments: Schema.Array(AddressedSequenceSegmentIndexSchema)
}).pipe(
  Schema.filter(
    (index) => sequenceIndexIssue(index) === undefined,
    {
      message: () => 'invalid addressed sequence index'
    }
  )
)

export interface AddressedSequenceWindowPart {
  readonly segmentId: string
  readonly address: string
  readonly start: number
  readonly end: number
  readonly itemIds: readonly string[]
}

export interface AddressedSequence<Item extends AddressedSequenceItem> {
  readonly empty: AddressedSequenceIndex
  /** Sentinel address representing this instance's index structure. */
  readonly sentinelAddress: string
  readonly append: (
    transaction: AddressSpaceTransaction<AddressedSequenceSegment<Item>>,
    index: AddressedSequenceIndex,
    item: Item
  ) => Effect.Effect<AddressedSequenceIndex, AddressedError>
  readonly updateById: (
    transaction: AddressSpaceTransaction<AddressedSequenceSegment<Item>>,
    index: AddressedSequenceIndex,
    itemId: string,
    update: (item: Item) => Item
  ) => Effect.Effect<AddressedSequenceIndex, AddressedError>
  readonly removeById: (
    transaction: AddressSpaceTransaction<AddressedSequenceSegment<Item>>,
    index: AddressedSequenceIndex,
    itemId: string
  ) => Effect.Effect<AddressedSequenceIndex, AddressedError>
  readonly replaceAll: (
    transaction: AddressSpaceTransaction<AddressedSequenceSegment<Item>>,
    index: AddressedSequenceIndex,
    items: readonly Item[]
  ) => Effect.Effect<AddressedSequenceIndex, AddressedError>
  readonly replaceRange: (
    transaction: AddressSpaceTransaction<AddressedSequenceSegment<Item>>,
    index: AddressedSequenceIndex,
    start: number,
    end: number,
    items: readonly Item[]
  ) => Effect.Effect<AddressedSequenceIndex, AddressedError>
  readonly positionOfItem: (
    index: AddressedSequenceIndex,
    itemId: string
  ) => Option.Option<number>
  readonly resolveAddressForItem: (
    index: AddressedSequenceIndex,
    itemId: string
  ) => Option.Option<string>
  readonly resolveRangeWindow: (
    index: AddressedSequenceIndex,
    start: number,
    limit: number
  ) => readonly AddressedSequenceWindowPart[]
  readonly resolveTailWindow: (
    index: AddressedSequenceIndex,
    limit: number
  ) => readonly AddressedSequenceWindowPart[]
  readonly readAll: (
    index: AddressedSequenceIndex
  ) => Effect.Effect<readonly Item[], AddressedError>
  readonly readAllInTransaction: (
    transaction: AddressSpaceTransaction<AddressedSequenceSegment<Item>>,
    index: AddressedSequenceIndex
  ) => Effect.Effect<readonly Item[], AddressedError>
  readonly readWindow: (
    window: readonly AddressedSequenceWindowPart[]
  ) => Effect.Effect<readonly Item[], AddressedError>
  readonly readWindowInTransaction: (
    transaction: AddressSpaceTransaction<AddressedSequenceSegment<Item>>,
    window: readonly AddressedSequenceWindowPart[]
  ) => Effect.Effect<readonly Item[], AddressedError>
}

export interface AddressedSequenceOptions<Item extends AddressedSequenceItem> {
  readonly prefix: string
  readonly runtime: AddressSpaceRuntime<AddressedSequenceSegment<Item>>
}

const segmentId = (number: number): string =>
  `seg-${number}`

const missingSegment = (
  collection: string,
  address: string,
  operation: string
): AddressedCollectionError =>
  new AddressedCollectionError({
    collection,
    address,
    operation,
    reason: 'index referenced a segment that was absent from the entry store'
  })

const missingItem = (
  collection: string,
  itemId: string,
  operation: string
): AddressedCollectionError =>
  new AddressedCollectionError({
    collection,
    operation,
    reason: `index did not contain item "${itemId}"`
  })

const invalidRange = (
  collection: string,
  start: number,
  end: number,
  totalCount: number,
  operation: string
): AddressedCollectionError =>
  new AddressedCollectionError({
    collection,
    operation,
    reason: `invalid range ${start}..${end} for sequence of ${totalCount} item(s)`
  })

const shortSegment = (
  collection: string,
  address: string,
  expected: number,
  actual: number,
  operation: string
): AddressedCollectionError =>
  new AddressedCollectionError({
    collection,
    address,
    operation,
    reason: `index expected ${expected} item(s), but segment contained ${actual}`
  })

const invalidSegmentIndex = (
  collection: string,
  index: AddressedSequenceSegmentIndex,
  operation: string,
  reason: string
): AddressedCollectionError =>
  new AddressedCollectionError({
    collection,
    address: index.address,
    operation,
    reason
  })

const invalidSequenceIndex = (
  collection: string,
  operation: string,
  reason: string
): AddressedCollectionError =>
  new AddressedCollectionError({
    collection,
    operation,
    reason
  })

const invalidWindowPart = (
  collection: string,
  part: AddressedSequenceWindowPart,
  operation: string,
  reason: string
): AddressedCollectionError =>
  new AddressedCollectionError({
    collection,
    address: part.address,
    operation,
    reason
  })

const failCollection = (
  error: AddressedCollectionError
): Effect.Effect<never, AddressedCollectionError> =>
  Effect.fail(error)

const replaceAt = <A>(items: readonly A[], index: number, value: A): readonly A[] =>
  items.map((item, itemIndex) => itemIndex === index ? value : item)

const replaceSegment = (
  index: AddressedSequenceIndex,
  segment: AddressedSequenceSegmentIndex
): AddressedSequenceIndex => ({
  ...index,
  segments: index.segments.map((candidate) =>
    candidate.id === segment.id ? segment : candidate
  )
})

const chunksOf = <A>(items: readonly A[], size: number): readonly (readonly A[])[] => {
  const chunks: A[][] = []
  for (let start = 0; start < items.length; start += size) {
    chunks.push(items.slice(start, start + size))
  }
  return chunks
}

const firstAffectedSegmentIndex = (
  index: AddressedSequenceIndex,
  start: number
): number => {
  const containing = index.segments.findIndex((segment) =>
    start < segment.start + segment.count
  )
  return containing === -1 ? index.segments.length : containing
}

const indexedSegmentItems = <Item extends AddressedSequenceItem>(
  collection: string,
  segment: AddressedSequenceSegment<Item>,
  index: AddressedSequenceSegmentIndex,
  operation: string
): Effect.Effect<readonly Item[], AddressedCollectionError> => {
  if (!Number.isInteger(index.start) || index.start < 0) {
    return Effect.fail(
      invalidSegmentIndex(collection, index, operation, `segment start must be a non-negative integer; got ${index.start}`)
    )
  }

  if (!Number.isInteger(index.count) || index.count < 0) {
    return Effect.fail(
      invalidSegmentIndex(collection, index, operation, `segment count must be a non-negative integer; got ${index.count}`)
    )
  }

  if (index.itemIds.length !== index.count) {
    return Effect.fail(
      invalidSegmentIndex(collection, index, operation, `segment count ${index.count} did not match ${index.itemIds.length} item id(s)`)
    )
  }

  if (segment.items.length < index.count) {
    return Effect.fail(shortSegment(collection, index.address, index.count, segment.items.length, operation))
  }

  const items = segment.items.slice(0, index.count)
  for (const [itemIndex, item] of items.entries()) {
    const expectedId = index.itemIds[itemIndex]
    if (item.id !== expectedId) {
      return Effect.fail(
        invalidSegmentIndex(collection, index, operation, `index expected item "${expectedId}" at offset ${itemIndex}, but segment contained "${item.id}"`)
      )
    }
  }

  return Effect.succeed(items)
}

const windowSegmentItems = <Item extends AddressedSequenceItem>(
  collection: string,
  segment: AddressedSequenceSegment<Item>,
  part: AddressedSequenceWindowPart,
  operation: string
): Effect.Effect<readonly Item[], AddressedCollectionError> => {
  if (!Number.isInteger(part.start) || !Number.isInteger(part.end) || part.start < 0 || part.end < part.start) {
    return Effect.fail(
      invalidWindowPart(collection, part, operation, `invalid window ${part.start}..${part.end}`)
    )
  }

  if (part.itemIds.length !== part.end - part.start) {
    return Effect.fail(
      invalidWindowPart(collection, part, operation, `window expected ${part.end - part.start} item id(s), but carried ${part.itemIds.length}`)
    )
  }

  if (segment.items.length < part.end) {
    return Effect.fail(shortSegment(collection, part.address, part.end, segment.items.length, operation))
  }

  const items = segment.items.slice(part.start, part.end)
  for (const [index, item] of items.entries()) {
    const expectedId = part.itemIds[index]
    if (item.id !== expectedId) {
      return Effect.fail(
        invalidWindowPart(collection, part, operation, `window expected item "${expectedId}" at offset ${index}, but segment contained "${item.id}"`)
      )
    }
  }

  return Effect.succeed(items)
}

const validateSequenceIndex = (
  collection: string,
  index: AddressedSequenceIndex,
  operation: string
): Effect.Effect<void, AddressedCollectionError> => {
  const issue = sequenceIndexIssue(index)
  if (!issue) return Effect.void
  return Effect.fail(
    issue.segment
      ? invalidSegmentIndex(collection, issue.segment, operation, issue.reason)
      : invalidSequenceIndex(collection, operation, issue.reason)
  )
}

const validateNextSequenceIndex = (
  collection: string,
  index: AddressedSequenceIndex,
  operation: string
): Effect.Effect<AddressedSequenceIndex, AddressedCollectionError> =>
  Effect.as(validateSequenceIndex(collection, index, operation), index)

export const resolveAddressedSequenceRangeWindow = (
  index: AddressedSequenceIndex,
  start: number,
  limit: number
): readonly AddressedSequenceWindowPart[] => {
  if (limit <= 0 || start >= index.totalCount) return []
  const normalizedStart = Math.max(0, start)
  const end = Math.min(index.totalCount, normalizedStart + limit)
  const parts: AddressedSequenceWindowPart[] = []

  for (const segment of index.segments) {
    const segmentStart = segment.start
    const segmentEnd = segment.start + segment.count
    const overlapStart = Math.max(normalizedStart, segmentStart)
    const overlapEnd = Math.min(end, segmentEnd)
    if (overlapStart >= overlapEnd) continue
    parts.push({
      segmentId: segment.id,
      address: segment.address,
      start: overlapStart - segmentStart,
      end: overlapEnd - segmentStart,
      itemIds: segment.itemIds.slice(overlapStart - segmentStart, overlapEnd - segmentStart)
    })
  }

  return parts
}

export const resolveAddressedSequenceTailWindow = (
  index: AddressedSequenceIndex,
  limit: number
): readonly AddressedSequenceWindowPart[] => {
  if (limit <= 0) return []

  let remaining = limit
  const parts: AddressedSequenceWindowPart[] = []

  for (let i = index.segments.length - 1; i >= 0 && remaining > 0; i -= 1) {
    const segment = index.segments[i]
    const take = Math.min(remaining, segment.count)
    parts.push({
      segmentId: segment.id,
      address: segment.address,
      start: segment.count - take,
      end: segment.count,
      itemIds: segment.itemIds.slice(segment.count - take, segment.count)
    })
    remaining -= take
  }

  return parts.reverse()
}

export const positionOfAddressedSequenceItem = (
  index: AddressedSequenceIndex,
  itemId: string
): Option.Option<number> => {
  for (const segment of index.segments) {
    const offset = segment.itemIds.indexOf(itemId)
    if (offset !== -1) return Option.some(segment.start + offset)
  }
  return Option.none()
}

export const makeAddressedSequence = <Item extends AddressedSequenceItem>(
  options: AddressedSequenceOptions<Item>
): AddressedSequence<Item> => {
  const allocatedEntryAddress = (number: number) => childAddress(options.prefix, 'entries', entryId(number))

  const allocateAddress = (
    nextAddressNumber: number
  ): readonly [address: string, nextAddressNumber: number] => [
    allocatedEntryAddress(nextAddressNumber),
    nextAddressNumber + 1
  ]

  const canReuseAddressForItems = (
    segment: AddressedSequenceSegmentIndex,
    items: readonly Item[]
  ): boolean =>
    // Existing windows carry item ids and offsets. Reuse a physical entry only
    // when every previously indexed item keeps the same offset in the rewrite.
    items.length >= segment.count &&
    segment.itemIds.every((itemId, index) => items[index]?.id === itemId)

  const addressForItems = (
    segment: AddressedSequenceSegmentIndex | undefined,
    items: readonly Item[],
    nextAddressNumber: number
  ): readonly [address: string, nextAddressNumber: number] =>
    segment && canReuseAddressForItems(segment, items)
      ? [segment.address, nextAddressNumber]
      : allocateAddress(nextAddressNumber)

  const readSegment = (
    address: string,
    operation: string
  ): Effect.Effect<AddressedSequenceSegment<Item>, AddressedError> =>
    Effect.flatMap(
      options.runtime.get(address),
      (segment) =>
        Option.isSome(segment)
          ? Effect.succeed(segment.value)
          : Effect.fail(missingSegment(options.prefix, address, operation))
    )

  const readTransactionSegment = (
    transaction: AddressSpaceTransaction<AddressedSequenceSegment<Item>>,
    address: string,
    operation: string
  ): Effect.Effect<AddressedSequenceSegment<Item>, AddressedError> =>
    Effect.flatMap(
      transaction.get(address),
      (segment) =>
        Option.isSome(segment)
          ? Effect.succeed(segment.value)
          : Effect.fail(missingSegment(options.prefix, address, operation))
    )

  const replaceItems = (
    transaction: AddressSpaceTransaction<AddressedSequenceSegment<Item>>,
    index: AddressedSequenceIndex,
    start: number,
    end: number,
    items: readonly Item[],
    operation: 'replaceAll' | 'replaceRange' | 'removeById'
  ): Effect.Effect<AddressedSequenceIndex, AddressedError> =>
    Effect.gen(function* () {
      yield* validateSequenceIndex(options.prefix, index, operation)
      if (
        !Number.isInteger(start) ||
        !Number.isInteger(end) ||
        start < 0 ||
        end < start ||
        end > index.totalCount
      ) {
        return yield* failCollection(invalidRange(options.prefix, start, end, index.totalCount, operation))
      }

      const firstAffected = firstAffectedSegmentIndex(index, start)
      const prefixSegments = index.segments.slice(0, firstAffected)
      const affectedSegments = index.segments.slice(firstAffected)
      const affectedStart = affectedSegments[0]?.start ?? index.totalCount

      const affectedItems: Item[] = []
      for (const segment of affectedSegments) {
        const segmentValue = yield* readTransactionSegment(transaction, segment.address, operation)
        const segmentItems = yield* indexedSegmentItems(options.prefix, segmentValue, segment, operation)
        affectedItems.push(...segmentItems)
      }

      const localStart = start - affectedStart
      const localEnd = end - affectedStart
      const nextAffectedItems = [
        ...affectedItems.slice(0, localStart),
        ...items,
        ...affectedItems.slice(localEnd)
      ]

      const nextSegments: AddressedSequenceSegmentIndex[] = [...prefixSegments]
      let nextStart = affectedStart
      let nextAddressNumber = index.nextAddressNumber
      for (const [offset, chunk] of chunksOf(nextAffectedItems, ADDRESSED_SEQUENCE_SEGMENT_CAPACITY).entries()) {
        const previousSegment = affectedSegments[offset]
        const id = previousSegment?.id ?? segmentId(firstAffected + offset)
        const [address, allocatedNextAddressNumber] = addressForItems(previousSegment, chunk, nextAddressNumber)
        nextAddressNumber = allocatedNextAddressNumber
        yield* transaction.set(address, { items: chunk })
        nextSegments.push({
          id,
          address,
          start: nextStart,
          count: chunk.length,
          itemIds: chunk.map((item) => item.id)
        })
        nextStart += chunk.length
      }

      return yield* validateNextSequenceIndex(options.prefix, {
        nextSegmentNumber: nextSegments.length,
        nextAddressNumber,
        totalCount: index.totalCount - (end - start) + items.length,
        segments: nextSegments
      }, operation)
    })

  return {
    empty: {
      nextSegmentNumber: 0,
      nextAddressNumber: 0,
      totalCount: 0,
      segments: []
    },

    sentinelAddress: collectionSentinelAddress(options.prefix),

    append: (transaction, index, item) =>
      Effect.gen(function* () {
        yield* validateSequenceIndex(options.prefix, index, 'append')
        const tail = index.segments[index.segments.length - 1]

        if (!tail || tail.count >= ADDRESSED_SEQUENCE_SEGMENT_CAPACITY) {
          const id = segmentId(index.nextSegmentNumber)
          const [address, nextAddressNumber] = allocateAddress(index.nextAddressNumber)
          yield* transaction.set(address, { items: [item] })
          const segment: AddressedSequenceSegmentIndex = {
            id,
            address,
            start: index.totalCount,
            count: 1,
            itemIds: [item.id]
          }
          return yield* validateNextSequenceIndex(options.prefix, {
            nextSegmentNumber: index.nextSegmentNumber + 1,
            nextAddressNumber,
            totalCount: index.totalCount + 1,
            segments: [...index.segments, segment]
          }, 'append')
        }

        const currentSegment = yield* transaction.get(tail.address).pipe(
          Effect.flatMap((segment) =>
            Option.isSome(segment)
              ? Effect.succeed(segment.value)
              : Effect.fail(missingSegment(options.prefix, tail.address, 'append'))
          )
        )
        const currentItems = yield* indexedSegmentItems(options.prefix, currentSegment, tail, 'append')
        yield* transaction.set(tail.address, {
          items: [...currentItems, item]
        })

        return yield* validateNextSequenceIndex(options.prefix, replaceSegment(
          {
            ...index,
            totalCount: index.totalCount + 1
          },
          {
            ...tail,
            count: tail.count + 1,
            itemIds: [...tail.itemIds, item.id]
          }
        ), 'append')
      }),

    updateById: (transaction, index, itemId, update) =>
      Effect.gen(function* () {
        yield* validateSequenceIndex(options.prefix, index, 'updateById')
        const segment = index.segments.find((candidate) => candidate.itemIds.includes(itemId))
        if (!segment) return yield* failCollection(missingItem(options.prefix, itemId, 'updateById'))

        const currentSegment = yield* transaction.get(segment.address).pipe(
          Effect.flatMap((value) =>
            Option.isSome(value)
              ? Effect.succeed(value.value)
              : Effect.fail(missingSegment(options.prefix, segment.address, 'updateById'))
          )
        )
        const currentItems = yield* indexedSegmentItems(options.prefix, currentSegment, segment, 'updateById')
        const itemIndex = currentItems.findIndex((item) => item.id === itemId)
        if (itemIndex === -1) return yield* failCollection(missingItem(options.prefix, itemId, 'updateById'))

        const current = currentItems[itemIndex]
        const updated = update(current)
        if (updated === current) return index
        const nextItems = replaceAt(currentItems, itemIndex, updated)

        if (updated.id === current.id) {
          yield* transaction.set(segment.address, { items: nextItems })
          return index
        }

        const [address, nextAddressNumber] = allocateAddress(index.nextAddressNumber)
        yield* transaction.set(address, { items: nextItems })

        const replaced = replaceSegment(index, {
          ...segment,
          address,
          itemIds: replaceAt(segment.itemIds, itemIndex, updated.id)
        })
        return yield* validateNextSequenceIndex(options.prefix, {
          ...replaced,
          nextAddressNumber
        }, 'updateById')
      }),

    removeById: (transaction, index, itemId) =>
      Option.match(positionOfAddressedSequenceItem(index, itemId), {
        onNone: () => failCollection(missingItem(options.prefix, itemId, 'removeById')),
        onSome: (position) =>
          replaceItems(transaction, index, position, position + 1, [], 'removeById')
      }),

    replaceAll: (transaction, index, items) =>
      replaceItems(transaction, index, 0, index.totalCount, items, 'replaceAll'),

    replaceRange: (transaction, index, start, end, items) =>
      replaceItems(transaction, index, start, end, items, 'replaceRange'),

    positionOfItem: positionOfAddressedSequenceItem,

    resolveAddressForItem: (index, itemId) =>
      Option.fromNullable(
        index.segments.find((segment) => segment.itemIds.includes(itemId))?.address
      ),

    resolveRangeWindow: resolveAddressedSequenceRangeWindow,

    resolveTailWindow: resolveAddressedSequenceTailWindow,

    readAll: (index) =>
      Effect.gen(function* () {
        yield* validateSequenceIndex(options.prefix, index, 'readAll')
        const items: Item[] = []
        for (const segmentIndex of index.segments) {
          const segment = yield* readSegment(segmentIndex.address, 'readAll')
          const segmentItems = yield* indexedSegmentItems(options.prefix, segment, segmentIndex, 'readAll')
          items.push(...segmentItems)
        }
        return items
      }),

    readAllInTransaction: (transaction, index) =>
      Effect.gen(function* () {
        yield* validateSequenceIndex(options.prefix, index, 'readAll')
        const items: Item[] = []
        for (const segmentIndex of index.segments) {
          const segment = yield* readTransactionSegment(transaction, segmentIndex.address, 'readAll')
          const segmentItems = yield* indexedSegmentItems(options.prefix, segment, segmentIndex, 'readAll')
          items.push(...segmentItems)
        }
        return items
      }),

    readWindow: (window) =>
      Effect.gen(function* () {
        const items: Item[] = []
        for (const part of window) {
          const segment = yield* readSegment(part.address, 'readWindow')
          const segmentItems = yield* windowSegmentItems(options.prefix, segment, part, 'readWindow')
          items.push(...segmentItems)
        }
        return items
      }),

    readWindowInTransaction: (transaction, window) =>
      Effect.gen(function* () {
        const items: Item[] = []
        for (const part of window) {
          const segment = yield* readTransactionSegment(transaction, part.address, 'readWindow')
          const segmentItems = yield* windowSegmentItems(options.prefix, segment, part, 'readWindow')
          items.push(...segmentItems)
        }
        return items
      })
  }
}
