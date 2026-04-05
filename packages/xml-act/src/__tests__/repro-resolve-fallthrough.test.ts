import { describe, it, expect } from 'vitest'
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

const knownTags = new Set(['edit', 'shell', 'write', 'read'])
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
    // Model outputs <task id="t1"> inside the <old> child of an <edit> tool
    const xml = `<lenses>
<lens name="turn">planning</lens>
</lenses>
<task id="t1">
<edit path="foo.ts" observe=".">
<old>before <task id="t1"> middle </task> after</old>
<new>replaced</new>
</edit>
</task>`

    const events = parse(xml)

    // The edit should complete with the child containing the literal text
    const editClosed = events.filter(
      (e): e is Extract<ParseEvent, { _tag: 'TagClosed' }> =>
        e._tag === 'TagClosed' && e.tagName === 'edit',
    )
    expect(editClosed).toHaveLength(1)

    const oldChild = editClosed[0].element.children.find(c => c.tagName === 'old')
    expect(oldChild).toBeDefined()
    expect(oldChild!.body).toBe('before <task id="t1"> middle </task> after')

    // Should have exactly ONE TagOpened (the outer actions)
    const containerOpens = events.filter(e => e._tag === 'TagOpened')
    expect(containerOpens).toHaveLength(1)
  })

  it('think/lenses tags inside child body should be passthrough', () => {
    const xml = `<task id="t1">
<edit path="foo.ts" observe=".">
<old>code with <lenses> and </lenses> in it</old>
<new>replaced</new>
</edit>
</task>`

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
    const xml = `<task id="t1">
<edit path="foo.ts" observe=".">
<old>code mentioning <idle/> and <idle/> tags</old>
<new>replaced</new>
</edit>
</task>`

    const events = parse(xml)

    const editClosed = events.filter(
      (e): e is Extract<ParseEvent, { _tag: 'TagClosed' }> =>
        e._tag === 'TagClosed' && e.tagName === 'edit',
    )
    expect(editClosed).toHaveLength(1)

    const oldChild = editClosed[0].element.children.find(c => c.tagName === 'old')
    expect(oldChild).toBeDefined()
    expect(oldChild!.body).toContain('<idle/>')

    // Turn control should NOT have been emitted from inside the child body
    const turnControls = events.filter(e => e._tag === 'TurnControl')
    expect(turnControls).toHaveLength(0)
  })

  it('message tags inside child body should be passthrough', () => {
    const xml = `<task id="t1">
<edit path="foo.ts" observe=".">
<old>has <message>hello</message> in it</old>
<new>replaced</new>
</edit>
</task>`

    const events = parse(xml)

    const editClosed = events.filter(
      (e): e is Extract<ParseEvent, { _tag: 'TagClosed' }> =>
        e._tag === 'TagClosed' && e.tagName === 'edit',
    )
    expect(editClosed).toHaveLength(1)

    const oldChild = editClosed[0].element.children.find(c => c.tagName === 'old')
    expect(oldChild).toBeDefined()
    expect(oldChild!.body).toContain('<message>hello</message>')

    // No MessageStart should have been emitted
    const messageStarts = events.filter(e => e._tag === 'MessageStart')
    expect(messageStarts).toHaveLength(0)
  })

  it('other tool tags inside child body should be passthrough', () => {
    const xml = `<task id="t1">
<edit path="foo.ts" observe=".">
<old>code with <shell observe=".">echo hi</shell> in it</old>
<new>replaced</new>
</edit>
</task>`

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
    const xml = `<task id="t1">
<edit path="foo.ts" observe=".">
<old>before <task id="t1"> middle </task> after</old>
<new>replaced</new>
</edit>
</task>`

    const events = parseCharByChar(xml)

    const editClosed = events.filter(
      (e): e is Extract<ParseEvent, { _tag: 'TagClosed' }> =>
        e._tag === 'TagClosed' && e.tagName === 'edit',
    )
    expect(editClosed).toHaveLength(1)

    const oldChild = editClosed[0].element.children.find(c => c.tagName === 'old')
    expect(oldChild).toBeDefined()
    expect(oldChild!.body).toBe('before <task id="t1"> middle </task> after')
  })
})

describe('BUG 2: body-capture fallthrough — structural tags inside finish body', () => {
  it('structural tags inside finish body should be captured as text', () => {
    const xml = `<finish>The task is done. Used <task id="t1"> and <shell> to complete it.</finish>`

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
    const containerOpens = events.filter(e => e._tag === 'TagOpened')
    expect(containerOpens).toHaveLength(0)

    const toolOpens = events.filter(e => e._tag === 'TagOpened')
    expect(toolOpens).toHaveLength(0)
  })
})

describe('BUG 3: lenses with no active lens — structural tags after last lens', () => {
  it.skip('auto-closes lenses and handles structural tags normally', () => {
    const xml = `<lenses>
<lens name="intent">thinking about intent</lens>
<task id="t2">
<message>hello</message>
</task>`

    const events = parse(xml)

    const lensEnds = events.filter(e => e._tag === 'LensEnd')
    expect(lensEnds).toHaveLength(1)

    const commsOpens = events.filter(
      (e): e is Extract<ParseEvent, { _tag: 'TagOpened' }> =>
        e._tag === 'TagOpened',
    )
    expect(commsOpens).toHaveLength(1)

    const messageStarts = events.filter(e => e._tag === 'MessageStart')
    expect(messageStarts).toHaveLength(1)

    const parseErrors = events.filter(e => e._tag === 'ParseError')
    expect(parseErrors.length).toBeLessThanOrEqual(1)
  })
})
