import { describe, expect, it } from 'vitest'
import { createStreamingXmlParser } from '../parser'

describe('cross-batch coalescing', () => {
  it('does not lose text when MessageChunk spans multiple processChunk calls', () => {
    const parser = createStreamingXmlParser(
      new Set(['shell']),
      new Map(),
      undefined,
      () => 'test-id',
      'user',
    )

    const batch1 = parser.processChunk(
      '<lens name="turn">planning</lens>\n<task id="t2">\n<message>Hey Anders! What can',
    )

    const batch2 = parser.processChunk(
      ' I help you with today?</message>\n</task>\n',
    )

    const messageChunks = (events: readonly unknown[]) =>
      events
        .filter(
          (e): e is { _tag: 'MessageChunk'; text: string } =>
            typeof e === 'object' &&
            e !== null &&
            '_tag' in e &&
            (e as { _tag?: string })._tag === 'MessageChunk',
        )
        .map((e) => e.text)
        .join('')

    expect(messageChunks(batch1)).toContain('Hey Anders! What can')
    expect(messageChunks(batch2)).toContain(' I help you with today?')
    expect(messageChunks([...batch1, ...batch2])).toContain(
      'Hey Anders! What can I help you with today?',
    )
  })
})
