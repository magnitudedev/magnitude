import type { Projection } from '@magnitudedev/event-core'
import { Effect, Option } from 'effect'
import type { DisplayMessage } from '../types'

export type DisplayMessageSequence = Projection.ProjectionAddressedSequenceHandle<DisplayMessage>
export type DisplayMessageSequenceIndex = DisplayMessageSequence['empty']

export const appendDisplayMessage = (
  sequence: DisplayMessageSequence,
  index: DisplayMessageSequenceIndex,
  message: DisplayMessage
) =>
  sequence.append(index, message)

/** Read a single message by id through its containing segment only. */
export const readDisplayMessageById = (
  sequence: DisplayMessageSequence,
  index: DisplayMessageSequenceIndex,
  id: string
) =>
  Option.match(sequence.positionOfItem(index, id), {
    onNone: () => Effect.succeed(Option.none<DisplayMessage>()),
    onSome: (position) =>
      Effect.map(
        sequence.readWindow(sequence.resolveRangeWindow(index, position, 1)),
        (items) => Option.fromNullable(items[0])
      )
  })

/** Item ids of the sequence's tail suffix, resolved from the index alone. */
export const tailDisplayMessageIds = (
  sequence: DisplayMessageSequence,
  index: DisplayMessageSequenceIndex,
  count: number
): readonly string[] =>
  sequence.resolveTailWindow(index, count).flatMap((part) => part.itemIds)
