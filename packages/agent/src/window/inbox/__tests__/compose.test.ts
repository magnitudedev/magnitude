import { describe, expect, test } from 'vitest'
import { Option } from 'effect'
import {
  composeTimelineUserMessageItems,
  toTimelineAgentBlock,
  toTimelineObservation,
  toTimelineUserMessage,
} from '../compose'
import type { AgentAtom } from '../types'
import type { MentionOccurrence, MentionResolution } from '../../../events'

const TS = 1711641600000

describe('inbox compose', () => {
  test('preserves authored ordering by replacing inline mention spans with semantic items', () => {
    const text = 'before @image.png after'
    const occurrence: MentionOccurrence = {
      occurrenceId: 'mention-1',
      attachment: { type: 'mention_file', path: 'image.png' },
      placement: { _tag: 'inline', start: 7, end: 17 },
    }
    const resolution: MentionResolution = {
      occurrenceId: 'mention-1',
      status: 'resolved',
      parts: [{
        _tag: 'ContextImage',
        data: 'YWJj',
        mediaType: 'image/png',
        dimensions: { width: 1, height: 1 },
        path: 'image.png',
        name: Option.some('image.png'),
        byteSize: Option.some(3),
      }],
      truncated: false,
    }

    const items = composeTimelineUserMessageItems({
      text,
      mentions: [occurrence],
      resolutions: [resolution],
      attachments: [],
    })

    expect(items.map(item => item.kind)).toEqual(['body', 'mention', 'body'])
    expect(items[0]).toEqual({ kind: 'body', parts: [{ _tag: 'ContextText', text: 'before ' }] })
    expect(items[1]).toMatchObject({ kind: 'mention', mention: { occurrence, resolution: { status: 'resolved' } } })
    expect(items[2]).toEqual({ kind: 'body', parts: [{ _tag: 'ContextText', text: ' after' }] })
  })

  test('appends trailing mentions and direct image attachments after the authored body', () => {
    const occurrence: MentionOccurrence = {
      occurrenceId: 'mention-1',
      attachment: { type: 'mention_file', path: 'notes.txt' },
      placement: { _tag: 'trailing' },
    }
    const items = composeTimelineUserMessageItems({
      text: 'hello',
      mentions: [occurrence],
      resolutions: [{ occurrenceId: 'mention-1', status: 'resolved', parts: [{ _tag: 'ContextText', text: 'notes' }], truncated: false }],
      attachments: [{
        type: 'image',
        image: {
          _tag: 'ContextImage', data: 'YWJj', mediaType: 'image/png',
          dimensions: { width: 1, height: 1 }, path: '/tmp/image.png',
          name: Option.none(), byteSize: Option.some(3),
        },
      }],
    })
    expect(items.map(item => item.kind)).toEqual(['body', 'mention', 'attachment'])
  })

  test('constructors preserve semantic arrays by reference', () => {
    const items = [{ kind: 'body' as const, parts: [{ _tag: 'ContextText' as const, text: 'hello' }] }]
    const atoms: readonly AgentAtom[] = [{ kind: 'thought', timestamp: TS, text: 'thinking' }]
    const observationParts = [{ _tag: 'ContextText' as const, text: 'observed' }]
    const message = toTimelineUserMessage({ timestamp: TS, items, synthetic: Option.none() })
    const block = toTimelineAgentBlock({
      timestamp: TS,
      firstAtomTimestamp: TS,
      lastAtomTimestamp: TS,
      agentId: 'a1', role: 'engineer', status: 'working', atoms,
    })
    const observation = toTimelineObservation({ timestamp: TS, parts: observationParts })

    if (message.kind !== 'user_message' || block.kind !== 'agent_block' || observation.kind !== 'observation') {
      throw new Error('unexpected timeline constructor result')
    }
    expect(message.items).toBe(items)
    expect(block.atoms).toBe(atoms)
    expect(observation.parts).toBe(observationParts)
  })
})
