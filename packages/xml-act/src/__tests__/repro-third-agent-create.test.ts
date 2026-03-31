import { describe, expect, it } from 'bun:test'
import { createStreamingXmlParser } from '../parser'
import type { XmlActEvent } from '../format/types'

/**
 * Repro: When multiple parsers are created (parent + subagent), the second
 * parser creation used to corrupt the first parser's structuralTags map
 * because it was a module-level mutable singleton.
 *
 * This caused the third tool call in a turn to be parsed as prose instead
 * of as a tool tag.
 *
 * Fixed by making structuralTags instance-scoped in createXmlActFormat().
 */

const parentTags = new Set(['write', 'agent-create'])
const parentChildMap = new Map([
  ['agent-create', new Set(['title', 'message'])],
])

const subagentTags = new Set(['shell', 'read', 'write', 'edit'])
const subagentChildMap = new Map([
  ['shell', new Set(['stdin'])],
])

function tagOpened(events: readonly XmlActEvent[]) {
  return events.filter(e => e._tag === 'TagOpened')
}

function messageChunks(events: readonly XmlActEvent[]) {
  return events.filter(e => e._tag === 'MessageChunk')
}

describe('repro: structuralTags singleton corruption across parsers', () => {
  it('subagent parser creation does not corrupt parent parser', () => {
    const parent = createStreamingXmlParser(parentTags, parentChildMap, undefined, undefined, 'user')

    const part1 = [
      '<lenses>',
      '<lens name="intent">planning</lens>',
      '</lenses>',
      '<comms>',
      '<message to="user">Implementing now.</message>',
      '</comms>',
      '<actions>',
      '<agent-create agentId="builder-1" type="builder" observe=".">',
      '<title>Unit 1</title>',
      '<message>Do thing 1</message>',
      '</agent-create>',
      '<agent-create agentId="builder-2" type="builder" observe=".">',
      '<title>Unit 2</title>',
      '<message>Do thing 2</message>',
      '</agent-create>',
    ].join('\n')

    const events1 = parent.processChunk(part1)
    expect(tagOpened(events1)).toHaveLength(2)

    // Simulate subagent parser creation (different tool set)
    createStreamingXmlParser(subagentTags, subagentChildMap, undefined, undefined, 'user')

    const part2 = [
      '<agent-create agentId="builder-3" type="builder" observe=".">',
      '<title>Unit 3</title>',
      '<message>Do thing 3</message>',
      '</agent-create>',
      '</actions>',
      '<yield/>',
    ].join('\n')

    const events2 = parent.processChunk(part2)
    expect(tagOpened(events2)).toHaveLength(1)
    expect(messageChunks(events2).some(e => (e as any).text.includes('builder-3'))).toBe(false)

    const all = [...events1, ...events2, ...parent.flush()]
    expect(tagOpened(all)).toHaveLength(3)
  })

  it('multiple subagent parsers do not interfere with each other', () => {
    const p1 = createStreamingXmlParser(parentTags, parentChildMap, undefined, undefined, 'user')
    const p2 = createStreamingXmlParser(subagentTags, subagentChildMap, undefined, undefined, 'user')
    const p3 = createStreamingXmlParser(parentTags, parentChildMap, undefined, undefined, 'user')

    const xml = '<actions>\n<agent-create agentId="x" type="builder" observe=".">\n<title>T</title>\n<message>M</message>\n</agent-create>\n</actions>\n<yield/>'

    const e1 = [...p1.processChunk(xml), ...p1.flush()]
    const e3 = [...p3.processChunk(xml), ...p3.flush()]

    expect(tagOpened(e1)).toHaveLength(1)
    expect(tagOpened(e3)).toHaveLength(1)
  })
})
