import { describe, expect, it } from 'vitest'
import { Effect } from 'effect'
import { Introspection } from '@magnitudedev/event-core'
import type { DisplayTimelineState } from '../src/display'
import {
  createAddressedAtlas,
  type AddressedAtlasNode,
  type AddressedAtlasSegment,
} from '../src/introspection/addressed-tree'

const displayTimelineState = (
  segments: DisplayTimelineState['messages']['segments'],
  totalCount: number,
): DisplayTimelineState => ({
  mode: 'idle',
  messages: {
    nextSegmentNumber: segments.length,
    nextAddressNumber: segments.length,
    totalCount,
    segments,
  },
  streamingMessageId: null,
  _currentTurnId: null,
  _pendingInboundCommunications: [],
  _queuedUserMessageCount: 0,
  _thinkingMessageId: null,
  _activeToolCallIds: [],
  _communicationMessageIdsByStreamId: {},
  _forkActivityMessageIdsByForkId: {},
})

const findSegment = (
  nodes: readonly AddressedAtlasNode[],
  id: string,
): AddressedAtlasSegment | null => {
  for (const node of nodes) {
    if (node.kind === 'segment' && node.logicalSegmentId === id) return node
    if (node.kind === 'group') {
      const child = findSegment(node.children, id)
      if (child) return child
    }
  }
  return null
}

describe('addressed atlas introspection', () => {
  it('builds one nested atlas with resident, offloaded, pinned, and unindexed addressed entries', async () => {
    const namespace = 'DisplayTimeline/messages'
    const statsByAddress = new Map([
      ['timeline/root/messages/entries/entry-0', 100],
      ['timeline/root/messages/entries/entry-1', 40],
    ])
    const registry: Introspection.AddressedIntrospectionRegistryService = {
      register: () => Effect.void,
      current: Effect.succeed([]),
      stats: (_namespace, addresses) =>
        Effect.succeed(
          [...addresses].flatMap((address) => {
            const storedBytes = statsByAddress.get(address)
            return storedBytes === undefined ? [] : [{ address, storedBytes }]
          })
        ),
    }

    const atlas = await Effect.runPromise(
      createAddressedAtlas(
        new Map([
          [null, displayTimelineState([
            {
              id: 'seg-0',
              address: 'timeline/root/messages/entries/entry-0',
              start: 0,
              count: 2,
              itemIds: ['message-a', 'message-b'],
            },
            {
              id: 'seg-1',
              address: 'timeline/root/messages/entries/entry-1',
              start: 2,
              count: 1,
              itemIds: ['message-c'],
            },
          ], 3)],
        ]),
        [{
          namespace,
          resident: [
            {
              address: 'timeline/root/messages/entries/entry-0',
              dirty: false,
              pinCount: 1,
              pinOwners: ['display-view:view-1'],
              estimatedBytes: 80,
            },
            {
              address: 'timeline/root/messages/entries/orphan',
              dirty: true,
              pinCount: 0,
              pinOwners: [],
              estimatedBytes: 13,
            },
          ],
          residentCount: 2,
          dirtyCount: 1,
          pinnedCount: 1,
          estimatedResidentBytes: 93,
          estimatedDirtyBytes: 13,
        }],
      ).pipe(
        Effect.provideService(Introspection.AddressedIntrospectionRegistry, registry)
      )
    )

    const root = atlas[0]
    expect(root).toMatchObject({
      kind: 'group',
      projection: 'DisplayTimeline',
      bytes: 133,
      residentEntryCount: 2,
      offloadedEntryCount: 1,
      pinnedEntryCount: 1,
      dirtyEntryCount: 1,
    })

    expect(findSegment(atlas, 'seg-0')).toMatchObject({
      residency: 'resident',
      estimatedResidentBytes: 80,
      estimatedStoredBytes: 100,
      estimatedBytes: 80,
      pins: [{ kind: 'display-view', viewId: 'view-1' }],
    })
    expect(findSegment(atlas, 'seg-1')).toMatchObject({
      residency: 'offloaded',
      estimatedResidentBytes: null,
      estimatedStoredBytes: 40,
      estimatedBytes: 40,
    })
    expect(JSON.stringify(atlas)).toContain('unindexed resident')
  })
})
