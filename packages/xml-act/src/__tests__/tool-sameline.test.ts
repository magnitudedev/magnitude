import { describe, expect, test } from 'vitest'
import { createStreamingXmlParser } from '../parser'

function parse(xml: string) {
  const knownTags = new Set(['skill', 'read', 'edit', 'create-task', 'spawn-worker', 'message'])
  const parser = createStreamingXmlParser(knownTags)
  parser.push(xml)
  parser.end()
  return parser.events
}

function toolTagNames(events: ReturnType<typeof parse>): string[] {
  return events
    .filter(e => e._tag === 'TagOpened')
    .map(e => (e as any).tagName ?? (e as any).tag ?? 'unknown')
}

describe('tool parsing on same line', () => {
  test('tool immediately after closing lens tag (no newline)', () => {
    const xml = `<lens name="skills">thinking</lens><skill name="bug" />`
    const events = parse(xml)
    const toolNames = toolTagNames(events)
    console.log('NO NEWLINE:', JSON.stringify(events.map(e => e._tag), null, 2))
    console.log('tool names:', toolNames)
    expect(toolNames).toContain('skill')
  })

  test('tool after closing lens tag with newline', () => {
    const xml = `<lens name="skills">thinking</lens>\n<skill name="bug" />`
    const events = parse(xml)
    const toolNames = toolTagNames(events)
    console.log('WITH NEWLINE:', JSON.stringify(events.map(e => e._tag), null, 2))
    expect(toolNames).toContain('skill')
  })

  test('tool after prose on same line', () => {
    const xml = `Now let me clean up the debug logs: <edit observe="." path="src/file.ts"><old>old code</old><new>new code</new></edit>`
    const events = parse(xml)
    const toolNames = toolTagNames(events)
    console.log('AFTER PROSE:', JSON.stringify(events.map(e => e._tag), null, 2))
    console.log('tool names:', toolNames)
    expect(toolNames).toContain('edit')
  })

  test('tool on its own line after prose', () => {
    const xml = `Now let me clean up the debug logs:\n<edit observe="." path="src/file.ts"><old>old code</old><new>new code</new></edit>`
    const events = parse(xml)
    const toolNames = toolTagNames(events)
    console.log('AFTER PROSE NEWLINE:', JSON.stringify(events.map(e => e._tag), null, 2))
    expect(toolNames).toContain('edit')
  })
})
