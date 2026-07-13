import { forkKeyToForkId, type ForkKey } from '@magnitudedev/protocol'
import { Addressed } from '@magnitudedev/event-core'
import type { DisplayTimelineWindowShape, DisplayViewShape } from './shape'

type AddressedSequenceIndex = Addressed.AddressedSequenceIndex
type AddressedSequenceWindowPart = Addressed.AddressedSequenceWindowPart

export interface DisplayTimelineDirectory {
  readonly messages: AddressedSequenceIndex
}

export interface ResolvedDisplayTimeline {
  readonly forkKey: ForkKey
  readonly forkId: string | null
  readonly shape: DisplayTimelineWindowShape
  readonly window: readonly AddressedSequenceWindowPart[]
  readonly addresses: ReadonlySet<string>
}

export interface DisplayViewAddressPlan {
  readonly acceptedShape: DisplayViewShape
  readonly timelines: readonly ResolvedDisplayTimeline[]
  readonly addresses: ReadonlySet<string>
}

export type DisplayTimelineDirectories = Readonly<Record<ForkKey, DisplayTimelineDirectory | undefined>>

const directoryFor = (
  directories: DisplayTimelineDirectories,
  forkKey: ForkKey
): DisplayTimelineDirectory | undefined =>
  directories[forkKey]

const addressesForWindow = (
  window: readonly AddressedSequenceWindowPart[]
): ReadonlySet<string> =>
  new Set(window.map((part) => part.address))

export const resolveTimelineWindow = (
  index: AddressedSequenceIndex,
  shape: DisplayTimelineWindowShape
): readonly AddressedSequenceWindowPart[] => {
  switch (shape.kind) {
    case 'tail':
      return Addressed.resolveAddressedSequenceTailWindow(index, shape.limit)
    case 'range':
      return Addressed.resolveAddressedSequenceRangeWindow(index, shape.start, shape.limit)
  }
}

export const resolveDisplayViewAddressPlan = (
  requestedShape: DisplayViewShape,
  directories: DisplayTimelineDirectories
): DisplayViewAddressPlan => {
  const timelines: ResolvedDisplayTimeline[] = []
  const acceptedTimelines: Record<ForkKey, DisplayTimelineWindowShape> = {}
  const addresses = new Set<string>()

  for (const [forkKey, shape] of Object.entries(requestedShape.timelines)) {
    const directory = directoryFor(directories, forkKey)
    if (!directory) continue

    const window = resolveTimelineWindow(directory.messages, shape)
    const timelineAddresses = addressesForWindow(window)
    for (const address of timelineAddresses) {
      addresses.add(address)
    }

    acceptedTimelines[forkKey] = shape
    timelines.push({
      forkKey,
      forkId: forkKeyToForkId(forkKey),
      shape,
      window,
      addresses: timelineAddresses
    })
  }

  return {
    acceptedShape: {
      timelines: acceptedTimelines
    },
    timelines,
    addresses
  }
}
