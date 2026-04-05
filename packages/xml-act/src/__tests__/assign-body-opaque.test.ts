import { describe, expect, test } from 'bun:test'
import { createStreamingXmlParser } from '../parser'
import type { ParseEvent } from '../format/types'

function parse(xml: string): ParseEvent[] {
  const parser = createStreamingXmlParser(new Set(), new Map())
  return [...parser.processChunk(xml), ...parser.flush()]
}

describe('assign body opaque capture (intentional red suite)', () => {
  test('unknown tags in assign body are preserved verbatim and emit no ProseChunk', () => {
    const events = parse(`<task id="t1" type="research" title="t">
<assign role="planner">
Unknown tags: <spawn>alpha</spawn> <start/> <deploy>beta</deploy>
</assign>
</task><idle/>`)

    const assign = events.find((e): e is Extract<ParseEvent, { _tag: 'TaskAssign' }> => e._tag === 'TaskAssign')
    expect(assign).toBeDefined()
    expect(assign?.role).toBe('planner')
    expect(assign?.body).toContain('<spawn>alpha</spawn>')
    expect(assign?.body).toContain('<start/>')
    expect(assign?.body).toContain('<deploy>beta</deploy>')

    const prose = events.filter((e): e is Extract<ParseEvent, { _tag: 'ProseChunk' }> => e._tag === 'ProseChunk')
    expect(prose).toHaveLength(0)
  })

  test('known structural tags in assign body stay literal and do not emit structural events', () => {
    const events = parse(`<task id="t1" type="research" title="t">
<assign role="planner">
<message to="user">should stay in assign body</message>
<task id="nested" type="plan" title="nope"></task>
<idle/>
</assign>
</task><idle/>`)

    const assign = events.find((e): e is Extract<ParseEvent, { _tag: 'TaskAssign' }> => e._tag === 'TaskAssign')
    expect(assign).toBeDefined()
    expect(assign?.body).toContain('<message to="user">should stay in assign body</message>')
    expect(assign?.body).toContain('<task id="nested" type="plan" title="nope"></task>')
    expect(assign?.body).toContain('<idle/>')

    const messageStarts = events.filter(
      (e): e is Extract<ParseEvent, { _tag: 'MessageStart' }> => e._tag === 'MessageStart',
    )
    expect(messageStarts).toHaveLength(0)

    const nestedTaskOpen = events.filter(
      (e): e is Extract<ParseEvent, { _tag: 'TaskOpen' }> => e._tag === 'TaskOpen' && e.id === 'nested',
    )
    expect(nestedTaskOpen).toHaveLength(0)

    const turnControls = events.filter(
      (e): e is Extract<ParseEvent, { _tag: 'TurnControl' }> => e._tag === 'TurnControl',
    )
    expect(turnControls).toHaveLength(1)
    expect(turnControls[0]?.decision).toBe('idle')
  })

  test('nested assign text is retained inside outer assign body', () => {
    const events = parse(`<task id="t1" type="research" title="t">
<assign role="planner">
outer start
<assign role="builder">inner payload</assign>
outer end
</assign>
</task><idle/>`)

    const assigns = events.filter((e): e is Extract<ParseEvent, { _tag: 'TaskAssign' }> => e._tag === 'TaskAssign')
    expect(assigns).toHaveLength(1)
    expect(assigns[0]?.body).toContain('<assign role="builder">inner payload</assign>')
    expect(assigns[0]?.body).toContain('outer start')
    expect(assigns[0]?.body).toContain('outer end')
  })

  test('unknown open/close tags in assign body never leak as prose', () => {
    const events = parse(`<task id="t1" type="research" title="t">
<assign role="planner">
mix <replace>r</replace> and <kill>k</kill>
</assign>
</task><idle/>`)

    const assign = events.find((e): e is Extract<ParseEvent, { _tag: 'TaskAssign' }> => e._tag === 'TaskAssign')
    expect(assign).toBeDefined()
    expect(assign?.body).toContain('<replace>r</replace>')
    expect(assign?.body).toContain('<kill>k</kill>')

    const prose = events.filter((e): e is Extract<ParseEvent, { _tag: 'ProseChunk' }> => e._tag === 'ProseChunk')
    expect(prose).toHaveLength(0)
  })
})
