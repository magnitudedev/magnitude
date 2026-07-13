import { describe, expect, it } from 'vitest'
import { Addressed } from '@magnitudedev/event-core'
import {
  resolveDisplayViewAddressPlan,
  resolveTimelineWindow,
  timelineRange,
  timelineTail,
  type DisplayTimelineDirectory,
  type DisplayViewShape
} from '../../src/display-view'

const sequenceIndex = (
  prefix: string,
  counts: readonly number[]
): Addressed.AddressedSequenceIndex => {
  let start = 0
  return {
    nextSegmentNumber: counts.length,
    nextAddressNumber: counts.length,
    totalCount: counts.reduce((sum, count) => sum + count, 0),
    segments: counts.map((count, index) => {
      const id = `seg-${index}`
      const segment = {
        id,
        address: `${prefix}/entries/entry-${index}`,
        start,
        count,
        itemIds: Array.from({ length: count }, (_, itemIndex) => `${id}:${itemIndex}`)
      }
      start += count
      return segment
    })
  }
}

describe('display view resolver', () => {
  it('resolves tail shapes to minimal addressed sequence windows', () => {
    const index = sequenceIndex('DisplayTimeline/messages/forks/root', [50, 50, 20])

    const window = resolveTimelineWindow(index, timelineTail(55))

    expect(window).toEqual([
      {
        segmentId: 'seg-1',
        address: 'DisplayTimeline/messages/forks/root/entries/entry-1',
        start: 15,
        end: 50,
        itemIds: Array.from({ length: 35 }, (_, index) => `seg-1:${index + 15}`)
      },
      {
        segmentId: 'seg-2',
        address: 'DisplayTimeline/messages/forks/root/entries/entry-2',
        start: 0,
        end: 20,
        itemIds: Array.from({ length: 20 }, (_, index) => `seg-2:${index}`)
      }
    ])
  })

  it('resolves range shapes to minimal addressed sequence windows', () => {
    const index = sequenceIndex('DisplayTimeline/messages/forks/root', [50, 50, 20])

    const window = resolveTimelineWindow(index, timelineRange(48, 5))

    expect(window).toEqual([
      {
        segmentId: 'seg-0',
        address: 'DisplayTimeline/messages/forks/root/entries/entry-0',
        start: 48,
        end: 50,
        itemIds: ['seg-0:48', 'seg-0:49']
      },
      {
        segmentId: 'seg-1',
        address: 'DisplayTimeline/messages/forks/root/entries/entry-1',
        start: 0,
        end: 3,
        itemIds: ['seg-1:0', 'seg-1:1', 'seg-1:2']
      }
    ])
  })

  it('builds an accepted address plan from requested shape and known directories', () => {
    const requestedShape: DisplayViewShape = {
      timelines: {
        root: timelineTail(75),
        'worker-a': timelineTail(5),
        missing: timelineTail(10)
      }
    }
    const directories: Record<string, DisplayTimelineDirectory> = {
      root: {
        messages: sequenceIndex('DisplayTimeline/messages/forks/root', [50, 50, 20])
      },
      'worker-a': {
        messages: sequenceIndex('DisplayTimeline/messages/forks/worker-a', [3])
      }
    }

    const plan = resolveDisplayViewAddressPlan(requestedShape, directories)

    expect(plan.acceptedShape).toEqual({
      timelines: {
        root: timelineTail(75),
        'worker-a': timelineTail(5)
      }
    })
    expect(plan.timelines.map((timeline) => [timeline.forkKey, timeline.forkId])).toEqual([
      ['root', null],
      ['worker-a', 'worker-a']
    ])
    expect([...plan.addresses].sort()).toEqual([
      'DisplayTimeline/messages/forks/root/entries/entry-0',
      'DisplayTimeline/messages/forks/root/entries/entry-1',
      'DisplayTimeline/messages/forks/root/entries/entry-2',
      'DisplayTimeline/messages/forks/worker-a/entries/entry-0'
    ].sort())
  })
})
