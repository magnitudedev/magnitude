import { describe, expect, test } from 'vitest'
import { createStreamingXmlParser } from '../index'
import type { ParseEvent } from '../index'
import { parseMarkdownToMdast } from '../../../../cli/src/markdown/parse'

function parse(xml: string): ParseEvent[] {
  const parser = createStreamingXmlParser(new Set(), new Map())
  return [...parser.processChunk(xml), ...parser.flush()]
}

function extractMessageText(xml: string): string {
  return parse(xml)
    .filter((event): event is Extract<ParseEvent, { _tag: 'MessageChunk' }> => event._tag === 'MessageChunk')
    .map(event => event.text)
    .join('')
}

function collectNodeTypes(node: any, out: string[] = []): string[] {
  if (!node || typeof node !== 'object') return out
  if (typeof node.type === 'string') out.push(node.type)
  for (const value of Object.values(node)) {
    if (Array.isArray(value)) {
      for (const child of value) collectNodeTypes(child, out)
    } else if (value && typeof value === 'object') {
      collectNodeTypes(value, out)
    }
  }
  return out
}

describe('message parser newline preservation', () => {
  test('preserves double newlines in message body', () => {
    const xml = `<task id="t2">
<message>line1

line2</message>
</task>
<idle/>`

    const events = parse(xml)
    const text = events
      .filter((event): event is Extract<ParseEvent, { _tag: 'MessageChunk' }> => event._tag === 'MessageChunk')
      .map(event => event.text)
      .join('')

    expect(text).toBe('line1\n\nline2')
    expect(text).toContain('\n\n')
  })

  test('paragraph before table merges when blank line is collapsed', () => {
    const xml = `<task id="t2">
<message>Some intro text

| Col1 | Col2 |
|------|------|
| a    | b    |</message>
</task>
<idle/>`

    const extracted = extractMessageText(xml)
    const doc = parseMarkdownToMdast(extracted)
    const nodeTypes = collectNodeTypes(doc)

    expect(extracted).toBe(`Some intro text

| Col1 | Col2 |
|------|------|
| a    | b    |`)
    expect(nodeTypes).toContain('table')
  })

  test('content after table merges into last row when blank line is collapsed', () => {
    const xml = `<task id="t2">
<message>| Col1 | Col2 |
|------|------|
| a    | b    |

This is a separate paragraph.</message>
</task>
<idle/>`

    const extracted = extractMessageText(xml)
    const doc = parseMarkdownToMdast(extracted)
    const topLevelTypes = (doc as any).children.map((node: any) => node.type)

    expect(extracted).toBe(`| Col1 | Col2 |
|------|------|
| a    | b    |

This is a separate paragraph.`)
    expect(topLevelTypes).toEqual(['table', 'paragraph'])
  })

  test("list followed by table doesn't parse as table", () => {
    const xml = `<task id="t2">
<message>- item 1
- item 2

| Col1 | Col2 |
|------|------|
| a    | b    |</message>
</task>
<idle/>`

    const extracted = extractMessageText(xml)
    const doc = parseMarkdownToMdast(extracted)
    const nodeTypes = collectNodeTypes(doc)

    expect(extracted).toBe(`- item 1
- item 2

| Col1 | Col2 |
|------|------|
| a    | b    |`)
    expect(nodeTypes).toContain('table')
  })
})
