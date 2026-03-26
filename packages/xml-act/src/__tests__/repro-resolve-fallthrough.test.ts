import { describe, it, expect } from 'bun:test'
import { createStreamingXmlParser } from '../parser'
import type { ParseEvent } from '../format/types'

/**
 * Repro: resolve() fallthrough bugs where tags inside opaque contexts
 * (child-body, body-capture, lenses-no-active-lens) fall through to
 * structural resolution instead of being treated as passthrough.
 *
 * These bugs were introduced by the format refactor (1c0594f) which
 * replaced per-frame candidate lists with a single resolve() function
 * that has a permissive default fallthrough.
 */

const knownTags = new Set(['edit', 'shell', 'fs-write', 'fs-read'])
const childTagMap = new Map<string, Set<string>>([
  ['edit', new Set(['old', 'new'])],
])

function parse(xml: string): ParseEvent[] {
  const parser = createStreamingXmlParser(knownTags, childTagMap)
  return [...parser.processChunk(xml), ...parser.flush()]
}

function parseCharByChar(xml: string): ParseEvent[] {
  const parser = createStreamingXmlParser(knownTags, childTagMap)
  const events: ParseEvent[] = []
  for (const ch of xml) {
    events.push(...parser.processChunk(ch))
  }
  events.push(...parser.flush())
  return events
}

describe('BUG 1: child-body fallthrough — structural tags inside tool child bodies', () => {
  it('structural container tag inside child body should be passthrough', () => {
    // Model outputs <actions> inside the <old> child of an <edit> tool
    const xml = `<lenses>
<lens name="turn">planning</lens>
</lenses>
<actions>
<edit path="foo.ts" observe=".">
<old>before <actions> middle </actions> after</old>
<new>replaced</new>
</edit>
</actions>`

    const events = parse(xml)

    // The edit should complete with the child containing the literal text
    const editClosed = events.filter(
      (e): e is Extract<ParseEvent, { _tag: 'TagClosed' }> =>
        e._tag === 'TagClosed' && e.tagName === 'edit',
    )
    expect(editClosed).toHaveLength(1)

    const oldChild = editClosed[0].element.children.find(c => c.tagName === 'old')
    expect(oldChild).toBeDefined()
    expect(oldChild!.body).toBe('before <actions> middle </actions> after')

    // Should have exactly ONE ContainerOpen (the outer actions)
    const containerOpens = events.filter(e => e._tag === 'ContainerOpen')
    expect(containerOpens).toHaveLength(1)
  })

  it('think/lenses tags inside child body should be passthrough', () => {
    const xml = `<actions>
<edit path="foo.ts" observe=".">
<old>code with <lenses> and </lenses> in it</old>
<new>replaced</new>
</edit>
</actions>`

    const events = parse(xml)

    const editClosed = events.filter(
      (e): e is Extract<ParseEvent, { _tag: 'TagClosed' }> =>
        e._tag === 'TagClosed' && e.tagName === 'edit',
    )
    expect(editClosed).toHaveLength(1)

    const oldChild = editClosed[0].element.children.find(c => c.tagName === 'old')
    expect(oldChild).toBeDefined()
    expect(oldChild!.body).toBe('code with <lenses> and </lenses> in it')
  })

  it('turn control tags inside child body should be passthrough', () => {
    const xml = `<actions>
<edit path="foo.ts" observe=".">
<old>code mentioning <next/> and <yield/> tags</old>
<new>replaced</new>
</edit>
</actions>`

    const events = parse(xml)

    const editClosed = events.filter(
      (e): e is Extract<ParseEvent, { _tag: 'TagClosed' }> =>
        e._tag === 'TagClosed' && e.tagName === 'edit',
    )
    expect(editClosed).toHaveLength(1)

    const oldChild = editClosed[0].element.children.find(c => c.tagName === 'old')
    expect(oldChild).toBeDefined()
    expect(oldChild!.body).toContain('<next/>')

    // Turn control should NOT have been emitted from inside the child body
    const turnControls = events.filter(e => e._tag === 'TurnControl')
    expect(turnControls).toHaveLength(0)
  })

  it('message tags inside child body should be passthrough', () => {
    const xml = `<actions>
<edit path="foo.ts" observe=".">
<old>has <message to="user">hello</message> in it</old>
<new>replaced</new>
</edit>
</actions>`

    const events = parse(xml)

    const editClosed = events.filter(
      (e): e is Extract<ParseEvent, { _tag: 'TagClosed' }> =>
        e._tag === 'TagClosed' && e.tagName === 'edit',
    )
    expect(editClosed).toHaveLength(1)

    const oldChild = editClosed[0].element.children.find(c => c.tagName === 'old')
    expect(oldChild).toBeDefined()
    expect(oldChild!.body).toContain('<message to="user">hello</message>')

    // No MessageStart should have been emitted
    const messageStarts = events.filter(e => e._tag === 'MessageStart')
    expect(messageStarts).toHaveLength(0)
  })

  it('other tool tags inside child body should be passthrough', () => {
    const xml = `<actions>
<edit path="foo.ts" observe=".">
<old>code with <shell observe=".">echo hi</shell> in it</old>
<new>replaced</new>
</edit>
</actions>`

    const events = parse(xml)

    const editClosed = events.filter(
      (e): e is Extract<ParseEvent, { _tag: 'TagClosed' }> =>
        e._tag === 'TagClosed' && e.tagName === 'edit',
    )
    expect(editClosed).toHaveLength(1)

    const oldChild = editClosed[0].element.children.find(c => c.tagName === 'old')
    expect(oldChild).toBeDefined()
    expect(oldChild!.body).toContain('<shell observe=".')

    // Shell should NOT have been opened as a separate tool
    const shellOpened = events.filter(
      (e): e is Extract<ParseEvent, { _tag: 'TagOpened' }> =>
        e._tag === 'TagOpened' && e.tagName === 'shell',
    )
    expect(shellOpened).toHaveLength(0)
  })

  it('char-by-char streaming: structural tags inside child body are passthrough', () => {
    const xml = `<actions>
<edit path="foo.ts" observe=".">
<old>before <actions> middle </actions> after</old>
<new>replaced</new>
</edit>
</actions>`

    const events = parseCharByChar(xml)

    const editClosed = events.filter(
      (e): e is Extract<ParseEvent, { _tag: 'TagClosed' }> =>
        e._tag === 'TagClosed' && e.tagName === 'edit',
    )
    expect(editClosed).toHaveLength(1)

    const oldChild = editClosed[0].element.children.find(c => c.tagName === 'old')
    expect(oldChild).toBeDefined()
    expect(oldChild!.body).toBe('before <actions> middle </actions> after')
  })
})

describe('BUG 2: body-capture fallthrough — structural tags inside finish body', () => {
  it('structural tags inside finish body should be captured as text', () => {
    const xml = `<finish>The task is done. Used <actions> and <shell> to complete it.</finish>`

    const events = parse(xml)

    const turnControl = events.find(
      (e): e is Extract<ParseEvent, { _tag: 'TurnControl'; decision: 'finish' }> =>
        e._tag === 'TurnControl' && e.decision === 'finish',
    )
    expect(turnControl).toBeDefined()
    if (!turnControl) return
    expect(turnControl.evidence).toContain('Used')
    expect(turnControl.evidence).toContain('complete it')

    // Should NOT have opened a container or tool
    const containerOpens = events.filter(e => e._tag === 'ContainerOpen')
    expect(containerOpens).toHaveLength(0)

    const toolOpens = events.filter(e => e._tag === 'TagOpened')
    expect(toolOpens).toHaveLength(0)
  })
})

describe('BUG 3: lenses with no active lens — structural tags after last lens', () => {
  it('auto-closes lenses and handles structural tags normally', () => {
    const xml = `<lenses>
<lens name="intent">thinking about intent</lens>
<comms>
<message to="user">hello</message>
</comms>`

    const events = parse(xml)

    const lensEnds = events.filter(e => e._tag === 'LensEnd')
    expect(lensEnds).toHaveLength(1)

    const commsOpens = events.filter(
      (e): e is Extract<ParseEvent, { _tag: 'ContainerOpen' }> =>
        e._tag === 'ContainerOpen' && e.tag === 'comms',
    )
    expect(commsOpens).toHaveLength(1)

    const messageStarts = events.filter(e => e._tag === 'MessageStart')
    expect(messageStarts).toHaveLength(1)

    const parseErrors = events.filter(e => e._tag === 'ParseError')
    expect(parseErrors).toHaveLength(0)
  })
})
