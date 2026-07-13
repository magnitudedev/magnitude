import { Effect } from 'effect'
import type { DisplayTimelineState } from '../display'
import {
  currentAddressedEntryStats,
  type AddressedResidentIntrospection,
  type AddressedSpaceIntrospection,
} from './addressed'

export interface AddressedPin {
  readonly owner: string
  readonly kind: 'display-view' | 'display-producer' | 'unknown'
  readonly viewId: string | null
  readonly producerId: string | null
}

export interface AddressedAtlasMetrics {
  readonly bytes: number
  readonly residentBytes: number
  readonly storedBytes: number
  readonly residentEntryCount: number
  readonly offloadedEntryCount: number
  readonly pinnedEntryCount: number
  readonly dirtyEntryCount: number
}

export interface AddressedAtlasGroup extends AddressedAtlasMetrics {
  readonly kind: 'group'
  readonly role: 'projection' | 'collection' | 'branch' | 'fork' | 'unindexed'
  readonly projection: string
  readonly label: string
  readonly path: readonly string[]
  readonly children: readonly AddressedAtlasNode[]
}

export interface AddressedAtlasSegment extends AddressedAtlasMetrics {
  readonly kind: 'segment'
  readonly projection: string
  readonly collection: string
  readonly namespace: string
  readonly address: string
  readonly label: string
  readonly path: readonly string[]
  readonly forkId: string | null
  readonly logicalSegmentId: string
  readonly startOffset: number
  readonly itemCount: number
  readonly itemIdsSample: readonly string[]
  readonly residency: 'resident' | 'offloaded'
  readonly dirty: boolean
  readonly estimatedResidentBytes: number | null
  readonly estimatedStoredBytes: number | null
  readonly estimatedBytes: number | null
  readonly pins: readonly AddressedPin[]
}

export interface AddressedAtlasResident extends AddressedAtlasMetrics {
  readonly kind: 'resident'
  readonly projection: string
  readonly namespace: string
  readonly address: string
  readonly label: string
  readonly path: readonly string[]
  readonly dirty: boolean
  readonly estimatedResidentBytes: number
  readonly pins: readonly AddressedPin[]
}

export type AddressedAtlasNode =
  | AddressedAtlasGroup
  | AddressedAtlasSegment
  | AddressedAtlasResident

const emptyMetrics = (): AddressedAtlasMetrics => ({
  bytes: 0,
  residentBytes: 0,
  storedBytes: 0,
  residentEntryCount: 0,
  offloadedEntryCount: 0,
  pinnedEntryCount: 0,
  dirtyEntryCount: 0,
})

const addMetrics = (
  left: AddressedAtlasMetrics,
  right: AddressedAtlasMetrics,
): AddressedAtlasMetrics => ({
  bytes: left.bytes + right.bytes,
  residentBytes: left.residentBytes + right.residentBytes,
  storedBytes: left.storedBytes + right.storedBytes,
  residentEntryCount: left.residentEntryCount + right.residentEntryCount,
  offloadedEntryCount: left.offloadedEntryCount + right.offloadedEntryCount,
  pinnedEntryCount: left.pinnedEntryCount + right.pinnedEntryCount,
  dirtyEntryCount: left.dirtyEntryCount + right.dirtyEntryCount,
})

const aggregateMetrics = (
  nodes: readonly AddressedAtlasNode[],
): AddressedAtlasMetrics =>
  nodes.reduce((metrics, node) => addMetrics(metrics, node), emptyMetrics())

const groupNode = (
  role: AddressedAtlasGroup['role'],
  projection: string,
  label: string,
  path: readonly string[],
  children: readonly AddressedAtlasNode[],
): AddressedAtlasGroup => ({
  kind: 'group',
  role,
  projection,
  label,
  path,
  children,
  ...aggregateMetrics(children),
})

export const addressedPin = (owner: string): AddressedPin => {
  if (owner.startsWith('display-view:')) {
    const [, viewId = null] = owner.split(':')
    return {
      owner,
      kind: 'display-view',
      viewId,
      producerId: null,
    }
  }

  if (owner.startsWith('DisplayTimeline:')) {
    const parts = owner.split(':')
    return {
      owner,
      kind: 'display-producer',
      viewId: null,
      producerId: parts.slice(2).join(':') || null,
    }
  }

  return {
    owner,
    kind: 'unknown',
    viewId: null,
    producerId: null,
  }
}

const residentNode = (
  projection: string,
  namespace: string,
  entry: AddressedResidentIntrospection,
  path: readonly string[],
): AddressedAtlasResident => {
  const pins = entry.pinOwners.map(addressedPin)
  return {
    kind: 'resident',
    projection,
    namespace,
    address: entry.address,
    label: entry.address,
    path,
    dirty: entry.dirty,
    estimatedResidentBytes: entry.estimatedBytes,
    pins,
    bytes: entry.estimatedBytes,
    residentBytes: entry.estimatedBytes,
    storedBytes: 0,
    residentEntryCount: 1,
    offloadedEntryCount: 0,
    pinnedEntryCount: pins.length > 0 ? 1 : 0,
    dirtyEntryCount: entry.dirty ? 1 : 0,
  }
}

export const createAddressedAtlas = (
  displayTimelineForks: ReadonlyMap<string | null, DisplayTimelineState>,
  spaces: readonly AddressedSpaceIntrospection[],
): Effect.Effect<readonly AddressedAtlasNode[]> =>
  Effect.gen(function* () {
    const projection = 'DisplayTimeline'
    const collection = 'messages'
    const namespace = `${projection}/${collection}`
    const space = spaces.find((candidate) => candidate.namespace === namespace)
    const residents = space?.resident ?? []
    const residentByAddress = new Map(residents.map((entry) => [entry.address, entry]))
    const segmentAddresses = [...new Set(
      [...displayTimelineForks.values()].flatMap((fork) =>
        fork.messages.segments.map((segment) => segment.address)
      )
    )]
    const storedStats = yield* currentAddressedEntryStats(namespace, segmentAddresses)
    const storedStatsByAddress = new Map(storedStats.map((stats) => [stats.address, stats]))
    const indexedAddresses = new Set<string>()

    const forkNodes = [...displayTimelineForks].map(([forkId, fork]) => {
      const forkLabel = forkId ?? 'root'
      const segmentNodes = fork.messages.segments.map((segment): AddressedAtlasSegment => {
        indexedAddresses.add(segment.address)
        const resident = residentByAddress.get(segment.address)
        const stored = storedStatsByAddress.get(segment.address)
        const pins = (resident?.pinOwners ?? []).map(addressedPin)
        const estimatedResidentBytes = resident?.estimatedBytes ?? null
        const estimatedStoredBytes = stored?.storedBytes ?? null
        const estimatedBytes = estimatedResidentBytes ?? estimatedStoredBytes
        const bytes = estimatedBytes ?? 0
        const residentBytes = estimatedResidentBytes ?? 0
        const storedBytes = estimatedStoredBytes ?? 0

        return {
          kind: 'segment',
          projection,
          collection,
          namespace,
          address: segment.address,
          label: segment.id,
          path: [projection, collection, 'forks', forkLabel, segment.id],
          forkId,
          logicalSegmentId: segment.id,
          startOffset: segment.start,
          itemCount: segment.count,
          itemIdsSample: segment.itemIds.slice(0, 8),
          residency: resident ? 'resident' : 'offloaded',
          dirty: resident?.dirty ?? false,
          estimatedResidentBytes,
          estimatedStoredBytes,
          estimatedBytes,
          pins,
          bytes,
          residentBytes,
          storedBytes,
          residentEntryCount: resident ? 1 : 0,
          offloadedEntryCount: resident ? 0 : 1,
          pinnedEntryCount: pins.length > 0 ? 1 : 0,
          dirtyEntryCount: resident?.dirty ? 1 : 0,
        }
      })

      return groupNode(
        'fork',
        projection,
        forkLabel,
        [projection, collection, 'forks', forkLabel],
        segmentNodes,
      )
    })

    const unindexedNodes = residents
      .filter((entry) => !indexedAddresses.has(entry.address))
      .map((entry) =>
        residentNode(
          projection,
          namespace,
          entry,
          [projection, collection, 'unindexed', entry.address],
        )
      )

    const collectionChildren: AddressedAtlasNode[] = [
      groupNode('branch', projection, 'forks', [projection, collection, 'forks'], forkNodes),
      ...(unindexedNodes.length > 0
        ? [groupNode('unindexed', projection, 'unindexed resident', [projection, collection, 'unindexed'], unindexedNodes)]
        : []),
    ]

    const root = groupNode(
      'projection',
      projection,
      projection,
      [projection],
      [
        groupNode(
          'collection',
          projection,
          collection,
          [projection, collection],
          collectionChildren,
        ),
      ],
    )

    return [root]
  })
