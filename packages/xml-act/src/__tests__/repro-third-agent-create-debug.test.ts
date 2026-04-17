import { describe, expect, it } from 'vitest'
import { createStreamingXmlParser } from '../parser'
import type { XmlActEvent } from '../format/types'

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

describe('debug: repro-third-agent-create', () => {
  it('shows what the 3 TagOpened events are', () => {
    const parent = createStreamingXmlParser(parentTags, parentChildMap, undefined, undefined, 'user')

    const part1 = [
      '<lens name="intent">planning</lens>',
      '<task id="t2">',
      '<message>Implementing now.</message>',
      '</task>',
      '<task id="t1">',
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
    const tags = tagOpened(events1)
    
    console.log('TagOpened events:', JSON.stringify(tags.map(t => ({
      tag: (t as any).tagName ?? (t as any).tag,
      id: (t as any).id ?? (t as any).toolId
    })), null, 2))
    
    // Expect 2 agent-create tags
    expect(tags).toHaveLength(2)
  })
  
  it('shows corruption when subagent is created (repro original issue)', () => {
    const parent = createStreamingXmlParser(parentTags, parentChildMap, undefined, undefined, 'user')

    const part1 = [
      '<lens name="intent">planning</lens>',
      '<task id="t2">',
      '<message>Implementing now.</message>',
      '</task>',
      '<task id="t1">',
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
    const tags = tagOpened(events1)
    
    // Simulate subagent parser creation (different tool set) - THIS IS THE CORRUPTION
    createStreamingXmlParser(subagentTags, subagentChildMap, undefined, undefined, 'user')

    const part2 = [
      '<agent-create agentId="builder-3" type="builder" observe=".">',
      '<title>Unit 3</title>',
      '<message>Do thing 3</message>',
      '</agent-create>',
      '</task>',
      '<idle/>',
    ].join('\n')

    const events2 = parent.processChunk(part2)
    const tags2 = tagOpened(events2)

    const all = [...events1, ...events2, ...parent.flush()]
    const allTags = tagOpened(all)
    
    console.log('All TagOpened events:', JSON.stringify(allTags.map(t => ({
      tag: (t as any).tagName ?? (t as any).tag,
      id: (t as any).id ?? (t as any).toolId
    })), null, 2))
    
    // This should be 2 but currently fails with 3 due to corruption
    expect(allTags).toHaveLength(2)
  })
})
