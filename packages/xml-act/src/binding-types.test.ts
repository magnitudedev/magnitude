/**
 * Binding Types & Integration Tests
 *
 * Verifies:
 * 1. Generic XmlBinding<T> constrains correctly for concrete T
 * 2. Erased XmlTagBinding (XmlBinding<unknown>) is usable at runtime
 * 3. childRecord.field maps correctly through the full pipeline
 * 4. childTags runtime parsing + input building
 * 5. Input builder produces correct output for all binding patterns
 */
import { describe, test, expect } from 'bun:test'
import { Effect, Stream } from 'effect'
import { Schema } from '@effect/schema'
import { createTool } from '@magnitudedev/tools'
import type { XmlBinding, XmlArrayChildBinding, InputFields } from '@magnitudedev/tools'
import {
  createXmlRuntime,
  buildInput,
  validateBinding,
  generateXmlToolDoc,
  type XmlRuntimeConfig,
  type XmlRuntimeEvent,
  type RegisteredTool,
  type XmlTagBinding,
  actionsTagOpen,
  actionsTagClose,
} from './index'

const ACTIONS_TAG_OPEN = actionsTagOpen()
const ACTIONS_TAG_CLOSE = actionsTagClose()
import type { ParsedElement } from './parser/types'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function runStream(cfg: XmlRuntimeConfig, xml: string): Promise<XmlRuntimeEvent[]> {
  const runtime = createXmlRuntime(cfg)
  const stream = runtime.streamWith(Stream.make(xml))
  return Effect.runPromise(Stream.runCollect(stream)).then(c => Array.from(c))
}

function reg(
  tool: ReturnType<typeof createTool>,
  tagName: string,
  binding: XmlTagBinding,
): RegisteredTool {
  return { tool, tagName, groupName: 'test', binding }
}

function cfg(tools: RegisteredTool[]): XmlRuntimeConfig {
  return { tools: new Map(tools.map(t => [t.tagName, t])) }
}

function ofType<T extends XmlRuntimeEvent['_tag']>(
  events: XmlRuntimeEvent[],
  tag: T,
): Extract<XmlRuntimeEvent, { _tag: T }>[] {
  return events.filter(e => e._tag === tag) as Extract<XmlRuntimeEvent, { _tag: T }>[]
}

function el(
  tagName: string,
  attrs: Record<string, string | number | boolean>,
  body: string,
  children: { tagName: string; attributes: Record<string, string | number | boolean>; body: string }[] = [],
): ParsedElement {
  return {
    tagName,
    toolCallId: 'tc_test',
    attributes: new Map(Object.entries(attrs)),
    body,
    children: children.map(c => ({
      tagName: c.tagName,
      attributes: new Map(Object.entries(c.attributes)),
      body: c.body,
    })),
  }
}

// =============================================================================
// 1. Compile-time type constraints
// =============================================================================

describe('type-level: XmlBinding<T> constraints', () => {
  type Input = {
    path: string
    content: string
    limit: number
    edits: Array<{ old: string; new: string }>
    vars: Record<string, string>
  }

  test('attributes constrained to InputFields<T>', () => {
    const _good: XmlBinding<Input> = {
      type: 'tag',
      attributes: [{ field: 'path', attr: 'path' }, { field: 'content', attr: 'content' }, { field: 'limit', attr: 'limit' }],
    }
    // @ts-expect-error — 'nonexistent' is not a field of Input
    const _bad: XmlBinding<Input> = { type: 'tag', attributes: [{ field: 'nonexistent', attr: 'nonexistent' }] }
    expect(true).toBe(true)
  })

  test('body constrained to InputFields<T>', () => {
    const _good: XmlBinding<Input> = { type: 'tag', body: 'content' }
    // @ts-expect-error — 'nonexistent' is not a field of Input
    const _bad: XmlBinding<Input> = { type: 'tag', body: 'nonexistent' }
    expect(true).toBe(true)
  })

  test('children field constrained to ArrayFields<T>', () => {
    const _good: XmlBinding<Input> = {
      type: 'tag',
      children: [{ field: 'edits', tag: 'edit', attributes: [{ field: 'old', attr: 'old' }], body: 'new' }],
    }
    // @ts-expect-error — 'path' is not an array field, so this child binding is invalid
    const _bad: XmlBinding<Input> = { type: 'tag', children: [{ field: 'path', tag: 'x' }] }
    expect(true).toBe(true)
  })

  test('children attributes constrained to element fields', () => {
    const _good: XmlBinding<Input> = {
      type: 'tag',
      children: [{ field: 'edits', attributes: [{ field: 'old', attr: 'old' }, { field: 'new', attr: 'new' }] }],
    }
    // @ts-expect-error — 'nonexistent' is not a field of the edits element type
    const _bad: XmlBinding<Input> = { type: 'tag', children: [{ field: 'edits', attributes: ['nonexistent'] }] }
    expect(true).toBe(true)
  })

  test('childRecord.field constrained to RecordFields<T>', () => {
    const _good: XmlBinding<Input> = {
      type: 'tag',
      childRecord: { field: 'vars', tag: 'var', keyAttr: 'name' },
    }
    // @ts-expect-error — 'path' is a string, not Record<string, string>
    const _bad: XmlBinding<Input> = { type: 'tag', childRecord: { field: 'path', tag: 'var', keyAttr: 'name' } }
    expect(true).toBe(true)
  })
})

describe('type-level: erased XmlTagBinding is usable', () => {
  test('XmlTagBinding accepts any string field names', () => {
    const binding: XmlTagBinding = {
      attributes: [{ field: 'anything', attr: 'anything' }, { field: 'goes', attr: 'goes' }],
      body: 'whatever',
      children: [{ field: 'stuff', tag: 'item', attributes: [{ field: 'a', attr: 'a' }, { field: 'b', attr: 'b' }], body: 'c' }],
      childRecord: { field: 'map', tag: 'entry', keyAttr: 'k' },
    }
    const attr: string = binding.attributes![0]!.field
    const body: string = binding.body!
    const childField: string = binding.children![0].field
    const recordField: string = binding.childRecord!.field
    expect(attr).toBe('anything')
    expect(body).toBe('whatever')
    expect(childField).toBe('stuff')
    expect(recordField).toBe('map')
  })

  test('XmlArrayChildBinding<unknown> falls back to XmlChildBinding', () => {
    const child: XmlArrayChildBinding<unknown> = {
      field: 'anything',
      tag: 'whatever',
      attributes: [{ field: 'any', attr: 'any' }, { field: 'string', attr: 'string' }],
      body: 'text',
    }
    const f: string = child.field
    expect(f).toBe('anything')
  })

  test('InputFields<unknown> is string', () => {
    const field: InputFields<unknown> = 'any_string_works'
    expect(field).toBe('any_string_works')
  })
})

// =============================================================================
// 2. childRecord.field — full pipeline
// =============================================================================

describe('childRecord.field end-to-end', () => {
  const envTool = createTool({
    name: 'set_env',
    description: 'Set environment variables',
    inputSchema: Schema.Struct({
      vars: Schema.Record({ key: Schema.String, value: Schema.String }),
    }),
    outputSchema: Schema.String,
    bindings: {
      xmlInput: { type: 'tag', childRecord: { field: 'vars', tag: 'var', keyAttr: 'name' } },
      xmlOutput: { type: 'tag' },
    } as const,
    execute: ({ vars }) => Effect.succeed(`set ${Object.keys(vars).join(',')}`),
  })

  const envBinding: XmlTagBinding = {
    childRecord: { field: 'vars', tag: 'var', keyAttr: 'name' },
  }

  test('binding validator accepts childRecord with field', () => {
    const schema = validateBinding('set_env', envBinding, envTool.inputSchema.ast)
    expect(schema.children.has('var')).toBe(true)
    const childSchema = schema.children.get('var')!
    expect(childSchema.attributes.has('name')).toBe(true)
    expect(childSchema.acceptsBody).toBe(true)
  })

  test('input builder maps to correct field name', () => {
    const element = el('set_env', {}, '', [
      { tagName: 'var', attributes: { name: 'FOO' }, body: 'bar' },
      { tagName: 'var', attributes: { name: 'BAZ' }, body: 'qux' },
    ])
    const input = buildInput(element, envBinding)
    expect(input).toEqual({ vars: { FOO: 'bar', BAZ: 'qux' } })
  })

  test('full runtime execution with childRecord.field', async () => {
    const xml = '<actions><set_env id="r1"><var name="A">1</var><var name="B">2</var></set_env></actions>'
    const events = await runStream(cfg([reg(envTool, 'set_env', envBinding)]), xml)

    const ready = ofType(events, 'ToolInputReady')
    expect(ready).toHaveLength(1)
    expect(ready[0].input).toEqual({ vars: { A: '1', B: '2' } })

    const ended = ofType(events, 'ToolExecutionEnded')
    expect(ended).toHaveLength(1)
    expect(ended[0].result._tag).toBe('Success')
    if (ended[0].result._tag === 'Success') {
      expect(ended[0].result.output).toBe('set A,B')
    }
  })

  test('childRecord.field mismatched with schema fails validation', () => {
    const badBinding: XmlTagBinding = {
      childRecord: { field: 'nonexistent', tag: 'var', keyAttr: 'name' },
    }
    expect(() => validateBinding('set_env', badBinding, envTool.inputSchema.ast)).toThrow(/does not exist/)
  })
})

// =============================================================================
// 3. childTags — runtime parsing + input building
// =============================================================================

describe('childTags runtime', () => {
  const agentTool = createTool({
    name: 'create',
    description: 'Create an agent',
    inputSchema: Schema.Struct({
      id: Schema.String,
      options: Schema.Struct({
        type: Schema.String,
        title: Schema.String,
        prompt: Schema.String,
      }),
    }),
    outputSchema: Schema.String,
    bindings: {
      xmlInput: { type: 'tag', attributes: [{ field: 'id', attr: 'id' }], childTags: [{ field: 'options.type', tag: 'type' }, { field: 'options.title', tag: 'title' }, { field: 'options.prompt', tag: 'prompt' }] },
      xmlOutput: { type: 'tag' },
    } as const,
    execute: ({ id, options }) => Effect.succeed(`created ${id}: ${options.type}`),
  })

  const agentBinding: XmlTagBinding = {
    attributes: [{ field: 'id', attr: 'id' }],
    childTags: [
      { field: 'options.type', tag: 'type' },
      { field: 'options.title', tag: 'title' },
      { field: 'options.prompt', tag: 'prompt' },
    ],
  }

  test('binding validator accepts childTags', () => {
    const schema = validateBinding('create', agentBinding, agentTool.inputSchema.ast)
    expect(schema.attributes.has('id')).toBe(true)
    expect(schema.children.has('type')).toBe(true)
    expect(schema.children.has('title')).toBe(true)
    expect(schema.children.has('prompt')).toBe(true)
  })

  test('input builder extracts childTag values', () => {
    const element = el('create', { id: 'agent-1' }, '', [
      { tagName: 'type', attributes: {}, body: 'builder' },
      { tagName: 'title', attributes: {}, body: 'Build the feature' },
      { tagName: 'prompt', attributes: {}, body: 'Implement X' },
    ])
    const input = buildInput(element, agentBinding)
    expect(input).toEqual({
      id: 'agent-1',
      options: {
        type: 'builder',
        title: 'Build the feature',
        prompt: 'Implement X',
      },
    })
  })

  test('full runtime execution with childTags', async () => {
    const xml = `<actions><create id="a1">
  <type>builder</type>
  <title>Build it</title>
  <prompt>Do the thing</prompt>
</create></actions>`
    const events = await runStream(cfg([reg(agentTool, 'create', agentBinding)]), xml)

    const ready = ofType(events, 'ToolInputReady')
    expect(ready).toHaveLength(1)
    expect(ready[0].input).toEqual({
      id: 'a1',
      options: {
        type: 'builder',
        title: 'Build it',
        prompt: 'Do the thing',
      },
    })

    const ended = ofType(events, 'ToolExecutionEnded')
    expect(ended).toHaveLength(1)
    expect(ended[0].result._tag).toBe('Success')
  })

  test('childTags with whitespace are trimmed', async () => {
    // Only leading/trailing newlines are stripped — interior whitespace is preserved
    // (important for content like code edits)
    const xml = `<actions><create id="a1">
  <type>
    builder
  </type>
  <title>  Build it  </title>
  <prompt>Do the thing</prompt>
</create></actions>`
    const events = await runStream(cfg([reg(agentTool, 'create', agentBinding)]), xml)

    const ready = ofType(events, 'ToolInputReady')
    expect(ready[0].input).toEqual({
      id: 'a1',
      options: {
        type: '    builder\n  ',
        title: '  Build it  ',
        prompt: 'Do the thing',
      },
    })
  })

  test('missing childTag is absent from input (not undefined)', async () => {
    const xml = `<actions><create id="a1"><type>builder</type></create></actions>`
    const events = await runStream(cfg([reg(agentTool, 'create', agentBinding)]), xml)

    const ready = ofType(events, 'ToolInputReady')
    expect(ready).toHaveLength(1)
    const input = ready[0].input as { id: string; options: Record<string, unknown> }
    expect(input.id).toBe('a1')
    expect(input.options.type).toBe('builder')
    expect('title' in input.options).toBe(false)
    expect('prompt' in input.options).toBe(false)
  })
})

// =============================================================================
// 4. Input builder — all binding patterns
// =============================================================================

describe('buildInput', () => {
  test('attributes only', () => {
    const input = buildInput(el('read', { path: 'src/index.ts' }, ''), { attributes: [{ field: 'path', attr: 'path' }] })
    expect(input).toEqual({ path: 'src/index.ts' })
  })

  test('body only', () => {
    const input = buildInput(el('shell', {}, '  ls -la  '), { body: 'command' })
    expect(input).toEqual({ command: 'ls -la' })
  })

  test('attributes + body', () => {
    const input = buildInput(
      el('write', { path: 'f.ts' }, 'content here'),
      { attributes: [{ field: 'path', attr: 'path' }], body: 'content' },
    )
    expect(input).toEqual({ path: 'f.ts', content: 'content here' })
  })

  test('children array binding', () => {
    const element = el('edit', { path: 'f.ts' }, '', [
      { tagName: 'change', attributes: { old: 'a' }, body: 'b' },
      { tagName: 'change', attributes: { old: 'c' }, body: 'd' },
    ])
    const input = buildInput(element, {
      attributes: [{ field: 'path', attr: 'path' }],
      children: [{ field: 'edits', tag: 'change', attributes: [{ field: 'old', attr: 'old' }], body: 'new' }],
    })
    expect(input).toEqual({
      path: 'f.ts',
      edits: [
        { old: 'a', new: 'b' },
        { old: 'c', new: 'd' },
      ],
    })
  })

  test('children uses field name as tag when tag not specified', () => {
    const element = el('batch', {}, '', [
      { tagName: 'items', attributes: { name: 'x' }, body: '' },
    ])
    const input = buildInput(element, {
      children: [{ field: 'items', attributes: [{ field: 'name', attr: 'name' }] }],
    })
    expect(input).toEqual({ items: [{ name: 'x' }] })
  })

  test('childTags binding', () => {
    const element = el('config', {}, '', [
      { tagName: 'host', attributes: {}, body: 'localhost' },
      { tagName: 'port', attributes: {}, body: '8080' },
    ])
    const input = buildInput(element, { childTags: [{ field: 'host', tag: 'host' }, { field: 'port', tag: 'port' }] })
    expect(input).toEqual({ host: 'localhost', port: '8080' })
  })

  test('childRecord binding with field', () => {
    const element = el('env', {}, '', [
      { tagName: 'var', attributes: { name: 'A' }, body: '1' },
      { tagName: 'var', attributes: { name: 'B' }, body: '2' },
    ])
    const input = buildInput(element, {
      childRecord: { field: 'vars', tag: 'var', keyAttr: 'name' },
    })
    expect(input).toEqual({ vars: { A: '1', B: '2' } })
  })

  test('childRecord tag name differs from field name', () => {
    const element = el('propose', { title: 'Plan' }, '', [
      { tagName: 'criterion', attributes: { id: 'c1' }, body: 'First criterion' },
      { tagName: 'criterion', attributes: { id: 'c2' }, body: 'Second criterion' },
    ])
    const input = buildInput(element, {
      attributes: [{ field: 'title', attr: 'title' }],
      childRecord: { field: 'criteria', tag: 'criterion', keyAttr: 'id' },
    })
    expect(input).toEqual({
      title: 'Plan',
      criteria: { c1: 'First criterion', c2: 'Second criterion' },
    })
  })

  test('empty binding produces empty input', () => {
    const input = buildInput(el('noop', {}, ''), {})
    expect(input).toEqual({})
  })

  test('unbound attributes are ignored', () => {
    const input = buildInput(
      el('read', { path: 'a.ts', extra: 'ignored' }, ''),
      { attributes: [{ field: 'path', attr: 'path' }] },
    )
    expect(input).toEqual({ path: 'a.ts' })
  })
})

// =============================================================================
// 5. Binding validation edge cases
// =============================================================================

describe('binding validation', () => {
  test('childTags referencing nonexistent field fails', () => {
    const tool = createTool({
      name: 'test', description: '',
      inputSchema: Schema.Struct({ name: Schema.String }),
      outputSchema: Schema.String,
      bindings: { xmlInput: { type: 'tag' }, xmlOutput: { type: 'tag' } } as const,
      execute: () => Effect.succeed(''),
    })
    expect(() => validateBinding('test', { childTags: [{ field: 'nonexistent', tag: 'nonexistent' }] }, tool.inputSchema.ast)).toThrow()
  })

  test('childRecord with nonexistent field fails', () => {
    const tool = createTool({
      name: 'test', description: '',
      inputSchema: Schema.Struct({ name: Schema.String }),
      outputSchema: Schema.String,
      bindings: { xmlInput: { type: 'tag' }, xmlOutput: { type: 'tag' } } as const,
      execute: () => Effect.succeed(''),
    })
    expect(() => validateBinding('test', {
      childRecord: { field: 'nonexistent', tag: 'x', keyAttr: 'k' },
    }, tool.inputSchema.ast)).toThrow(/does not exist/)
  })

  test('combined binding: attributes + childTags + childRecord', () => {
    const tool = createTool({
      name: 'complex', description: '',
      inputSchema: Schema.Struct({
        id: Schema.String,
        opts: Schema.Struct({
          mode: Schema.String,
          level: Schema.String,
        }),
        vars: Schema.Record({ key: Schema.String, value: Schema.String }),
      }),
      outputSchema: Schema.String,
      bindings: { xmlInput: { type: 'tag' }, xmlOutput: { type: 'tag' } } as const,
      execute: () => Effect.succeed(''),
    })
    const binding: XmlTagBinding = {
      attributes: [{ field: 'id', attr: 'id' }],
      childTags: [{ field: 'opts.mode', tag: 'mode' }, { field: 'opts.level', tag: 'level' }],
      childRecord: { field: 'vars', tag: 'var', keyAttr: 'name' },
    }
    const schema = validateBinding('complex', binding, tool.inputSchema.ast)
    expect(schema.attributes.has('id')).toBe(true)
    expect(schema.children.has('mode')).toBe(true)
    expect(schema.children.has('level')).toBe(true)
    expect(schema.children.has('var')).toBe(true)
  })
})

// =============================================================================
// 6. xml-docs uses childRecord.field for description lookup
// =============================================================================

describe('xml-docs childRecord.field integration', () => {
  test('childRecord docs use field description', () => {
    const tool = createTool({
      name: 'propose',
      description: 'Propose',
      inputSchema: Schema.Struct({
        title: Schema.String,
        criteria: Schema.optional(
          Schema.Record({ key: Schema.String, value: Schema.String })
        ).annotations({ description: 'Acceptance criteria map' }),
      }),
      outputSchema: Schema.Void,
      bindings: {
        xmlInput: {
          type: 'tag' as const,
          attributes: [{ field: 'title', attr: 'title' }] as const,
          childRecord: { field: 'criteria', tag: 'criterion', keyAttr: 'id' },
        },
        xmlOutput: { type: 'tag' as const },
      },
      execute: () => Effect.succeed(undefined),
    })
    const doc = generateXmlToolDoc(tool)!
    expect(doc).toContain('<criterion id="...">value</criterion>')
    expect(doc).toContain('Acceptance criteria map')
    expect(doc).toContain('<!-- ...more')
  })
})
