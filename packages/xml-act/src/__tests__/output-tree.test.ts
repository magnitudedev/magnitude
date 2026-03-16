import { describe, test, expect } from 'bun:test'
import { buildOutputTree, outputToText, outputToDOM, outputFromDOM, type OutputNode } from '../output-tree'
import type { XmlBinding } from '@magnitudedev/tools'
import { evaluateXPath } from 'fontoxpath'

// Shorthand: build + render in one step
function render(tagName: string, output: unknown, binding: XmlBinding<unknown>, echoAttrs?: Record<string, string>): string {
  return outputToText(buildOutputTree(tagName, output, binding, echoAttrs))
}

// Shorthand: build tree only
function build(tagName: string, output: unknown, binding: XmlBinding<unknown>, echoAttrs?: Record<string, string>): OutputNode {
  return buildOutputTree(tagName, output, binding, echoAttrs)
}

const TAG: XmlBinding<unknown> = { type: 'tag' }

// =============================================================================
// Scalar outputs
// =============================================================================

describe('buildOutputTree — scalar outputs', () => {
  test('string output', () => {
    expect(render('read', 'file contents', TAG)).toBe('<read>file contents</read>')
  })

  test('number output', () => {
    expect(render('add', 42, TAG)).toBe('<add>42</add>')
  })

  test('boolean output', () => {
    expect(render('check', true, TAG)).toBe('<check>true</check>')
  })

  test('string with special chars is NOT entity-encoded', () => {
    expect(render('read', 'x < y && a > b', TAG)).toBe('<read>x < y && a > b</read>')
  })

  test('string with angle brackets preserves them raw', () => {
    expect(render('read', 'const x = <T>(y: A & B) => {}', TAG))
      .toBe('<read>const x = <T>(y: A & B) => {}</read>')
  })

  test('string with quotes', () => {
    expect(render('read', 'say "hello"', TAG)).toBe('<read>say "hello"</read>')
  })

  test('scalar with echoAttrs', () => {
    expect(render('read', 'contents', TAG, { path: 'foo.ts' }))
      .toBe('<read path="foo.ts">contents</read>')
  })

  test('scalar with multiple echoAttrs', () => {
    expect(render('read', 'contents', TAG, { path: 'foo.ts', encoding: 'utf8' }))
      .toBe('<read path="foo.ts" encoding="utf8">contents</read>')
  })
})

// =============================================================================
// Void outputs
// =============================================================================

describe('buildOutputTree — void outputs', () => {
  test('undefined', () => {
    expect(render('write', undefined, TAG)).toBe('<write />')
  })

  test('null', () => {
    expect(render('write', null, TAG)).toBe('<write />')
  })

  test('void with echoAttrs', () => {
    expect(render('write', undefined, TAG, { path: 'out.ts' }))
      .toBe('<write path="out.ts" />')
  })
})

// =============================================================================
// childTags binding
// =============================================================================

describe('buildOutputTree — childTags', () => {
  const binding: XmlBinding<unknown> = {
    type: 'tag',
    childTags: [
      { field: 'content', tag: 'content' },
      { field: 'lines', tag: 'lines' },
    ],
  }

  test('struct with childTags', () => {
    const tree = build('read', { content: 'hello world', lines: 42 }, binding)
    expect(tree).toEqual({
      tag: 'element', name: 'read', attrs: {}, children: [
        { tag: 'element', name: 'content', attrs: {}, children: [{ tag: 'text', value: 'hello world' }] },
        { tag: 'element', name: 'lines', attrs: {}, children: [{ tag: 'text', value: '42' }] },
      ]
    })
  })

  test('childTags renders to text', () => {
    expect(render('read', { content: 'hello', lines: 10 }, binding))
      .toBe('<read><content>hello</content><lines>10</lines></read>')
  })

  test('childTags with special chars in values', () => {
    expect(render('shell', { stdout: 'a < b && c > d', exitCode: 0 }, {
      type: 'tag',
      childTags: [{ field: 'stdout', tag: 'stdout' }, { field: 'exitCode', tag: 'exitCode' }],
    })).toBe('<shell><stdout>a < b && c > d</stdout><exitCode>0</exitCode></shell>')
  })

  test('childTags skips undefined fields', () => {
    expect(render('read', { content: 'hello' }, binding))
      .toBe('<read><content>hello</content></read>')
  })

  test('childTags with nested field path', () => {
    expect(render('tool', { options: { mode: 'fast' } }, {
      type: 'tag',
      childTags: [{ field: 'options.mode', tag: 'mode' }],
    })).toBe('<tool><mode>fast</mode></tool>')
  })

  test('childTags with deeply nested field path', () => {
    expect(render('tool', { a: { b: { c: 'deep' } } }, {
      type: 'tag',
      childTags: [{ field: 'a.b.c', tag: 'value' }],
    })).toBe('<tool><value>deep</value></tool>')
  })

  test('childTags with missing nested field returns empty', () => {
    expect(render('tool', { a: {} }, {
      type: 'tag',
      childTags: [{ field: 'a.b.c', tag: 'value' }],
    })).toBe('<tool />')
  })
})

// =============================================================================
// body binding
// =============================================================================

describe('buildOutputTree — body', () => {
  test('body field only', () => {
    expect(render('write', { path: 'out.ts', content: 'hello' }, {
      type: 'tag',
      body: 'content',
    } as XmlBinding<unknown>)).toBe('<write>hello</write>')
  })

  test('body with attributes', () => {
    expect(render('write', { path: 'out.ts', content: 'hello' }, {
      type: 'tag',
      attributes: [{ field: 'path', attr: 'path' }],
      body: 'content',
    } as XmlBinding<unknown>)).toBe('<write path="out.ts">hello</write>')
  })

  test('body with special chars', () => {
    expect(render('write', { content: '<div class="x">&amp;</div>' }, {
      type: 'tag',
      body: 'content',
    } as XmlBinding<unknown>)).toBe('<write><div class="x">&amp;</div></write>')
  })
})

// =============================================================================
// attributes binding (on output)
// =============================================================================

describe('buildOutputTree — attributes', () => {
  test('output fields as attributes', () => {
    expect(render('result', { status: 'ok', code: 200 }, {
      type: 'tag',
      attributes: [{ field: 'status', attr: 'status' }, { field: 'code', attr: 'code' }],
    } as XmlBinding<unknown>)).toBe('<result status="ok" code="200" />')
  })

  test('attributes skip null/undefined', () => {
    expect(render('result', { status: 'ok', code: undefined }, {
      type: 'tag',
      attributes: [{ field: 'status', attr: 'status' }, { field: 'code', attr: 'code' }],
    } as XmlBinding<unknown>)).toBe('<result status="ok" />')
  })

  test('attributes combined with echoAttrs', () => {
    expect(render('result', { status: 'ok' }, {
      type: 'tag',
      attributes: [{ field: 'status', attr: 'status' }],
    } as XmlBinding<unknown>, { id: 'r1' })).toBe('<result id="r1" status="ok" />')
  })

  test('attributes + childTags', () => {
    expect(render('result', { status: 'ok', message: 'done' }, {
      type: 'tag',
      attributes: [{ field: 'status', attr: 'status' }],
      childTags: [{ field: 'message', tag: 'message' }],
    } as XmlBinding<unknown>)).toBe('<result status="ok"><message>done</message></result>')
  })
})

// =============================================================================
// items binding (direct array output)
// =============================================================================

describe('buildOutputTree — items', () => {
  test('array of structs with attributes only', () => {
    const output = [
      { file: 'a.ts', type: 'file', depth: 0 },
      { file: 'b/', type: 'dir', depth: 0 },
    ]
    expect(render('tree', output, {
      type: 'tag',
      items: { tag: 'entry', attributes: ['file', 'type', 'depth'] },
    } as XmlBinding<unknown>)).toBe('<tree><entry file="a.ts" type="file" depth="0" /><entry file="b/" type="dir" depth="0" /></tree>')
  })

  test('array of structs with attributes and body', () => {
    const output = [
      { file: 'a.ts', match: 'line 1: hello' },
      { file: 'b.ts', match: 'line 5: world' },
    ]
    expect(render('search', output, {
      type: 'tag',
      items: { tag: 'item', attributes: ['file'], body: 'match' },
    } as XmlBinding<unknown>)).toBe('<search><item file="a.ts">line 1: hello</item><item file="b.ts">line 5: world</item></search>')
  })

  test('array of scalars', () => {
    const output = ['foo', 'bar', 'baz']
    expect(render('list', output, {
      type: 'tag',
      items: { tag: 'item' },
    } as XmlBinding<unknown>)).toBe('<list><item>foo</item><item>bar</item><item>baz</item></list>')
  })

  test('array of numbers', () => {
    const output = [1, 2, 3]
    expect(render('nums', output, {
      type: 'tag',
      items: { tag: 'n' },
    } as XmlBinding<unknown>)).toBe('<nums><n>1</n><n>2</n><n>3</n></nums>')
  })

  test('empty array', () => {
    expect(render('list', [], {
      type: 'tag',
      items: { tag: 'item' },
    } as XmlBinding<unknown>)).toBe('<list />')
  })

  test('items with echoAttrs', () => {
    expect(render('search', [{ file: 'a.ts', match: 'hi' }], {
      type: 'tag',
      items: { tag: 'item', attributes: ['file'], body: 'match' },
    } as XmlBinding<unknown>, { pattern: '*.ts' })).toBe('<search pattern="*.ts"><item file="a.ts">hi</item></search>')
  })

  test('items with special chars in body', () => {
    expect(render('search', [{ file: 'a.ts', match: 'x < y && z > w' }], {
      type: 'tag',
      items: { tag: 'item', attributes: ['file'], body: 'match' },
    } as XmlBinding<unknown>)).toBe('<search><item file="a.ts">x < y && z > w</item></search>')
  })
})

// =============================================================================
// children binding (array fields as repeated child tags)
// =============================================================================

describe('buildOutputTree — children', () => {
  test('array field rendered as child elements with attributes', () => {
    const output = {
      results: [
        { path: 'a.ts', line: 10 },
        { path: 'b.ts', line: 20 },
      ]
    }
    expect(render('grep', output, {
      type: 'tag',
      children: [{ field: 'results', tag: 'match', attributes: [{ field: 'path', attr: 'path' }, { field: 'line', attr: 'line' }] }],
    } as XmlBinding<unknown>)).toBe('<grep><match path="a.ts" line="10" /><match path="b.ts" line="20" /></grep>')
  })

  test('children with body field', () => {
    const output = {
      entries: [
        { name: 'foo', description: 'A foo thing' },
        { name: 'bar', description: 'A bar thing' },
      ]
    }
    expect(render('help', output, {
      type: 'tag',
      children: [{ field: 'entries', tag: 'entry', attributes: [{ field: 'name', attr: 'name' }], body: 'description' }],
    } as XmlBinding<unknown>)).toBe('<help><entry name="foo">A foo thing</entry><entry name="bar">A bar thing</entry></help>')
  })

  test('children defaults tag to field name', () => {
    const output = {
      items: [{ id: '1' }, { id: '2' }]
    }
    expect(render('list', output, {
      type: 'tag',
      children: [{ field: 'items', attributes: [{ field: 'id', attr: 'id' }] }],
    } as XmlBinding<unknown>)).toBe('<list><items id="1" /><items id="2" /></list>')
  })

  test('children skips non-array field', () => {
    expect(render('list', { items: 'not an array' }, {
      type: 'tag',
      children: [{ field: 'items', tag: 'item', attributes: [{ field: 'id', attr: 'id' }] }],
    } as XmlBinding<unknown>)).toBe('<list />')
  })

  test('children with empty array', () => {
    expect(render('list', { items: [] }, {
      type: 'tag',
      children: [{ field: 'items', tag: 'item' }],
    } as XmlBinding<unknown>)).toBe('<list />')
  })
})

// =============================================================================
// childRecord binding
// =============================================================================

describe('buildOutputTree — childRecord', () => {
  test('record field as child elements', () => {
    const output = { vars: { HOME: '/home/user', PATH: '/usr/bin', SHELL: '/bin/bash' } }
    expect(render('env', output, {
      type: 'tag',
      childRecord: { field: 'vars', tag: 'var', keyAttr: 'name' },
    } as XmlBinding<unknown>)).toBe(
      '<env><var name="HOME">/home/user</var><var name="PATH">/usr/bin</var><var name="SHELL">/bin/bash</var></env>'
    )
  })

  test('empty record', () => {
    expect(render('env', { vars: {} }, {
      type: 'tag',
      childRecord: { field: 'vars', tag: 'var', keyAttr: 'name' },
    } as XmlBinding<unknown>)).toBe('<env />')
  })

  test('childRecord with special chars in keys and values', () => {
    const output = { meta: { 'content-type': 'text/html', 'x<y': 'a&b' } }
    const text = render('headers', output, {
      type: 'tag',
      childRecord: { field: 'meta', tag: 'header', keyAttr: 'key' },
    } as XmlBinding<unknown>)
    expect(text).toContain('key="content-type"')
    expect(text).toContain('a&b')
    expect(text).not.toContain('&amp;')
  })
})

// =============================================================================
// Combined bindings
// =============================================================================

describe('buildOutputTree — combined bindings', () => {
  test('attributes + childTags + body', () => {
    const output = { id: 'r1', status: 'ok', message: 'done', body: 'full details here' }
    expect(render('result', output, {
      type: 'tag',
      attributes: [{ field: 'id', attr: 'id' }, { field: 'status', attr: 'status' }],
      childTags: [{ field: 'message', tag: 'message' }],
      body: 'body',
    } as XmlBinding<unknown>)).toBe('<result id="r1" status="ok"><message>done</message>full details here</result>')
  })

  test('childTags + children', () => {
    const output = {
      total: 5,
      items: [{ name: 'a' }, { name: 'b' }],
    }
    expect(render('list', output, {
      type: 'tag',
      childTags: [{ field: 'total', tag: 'total' }],
      children: [{ field: 'items', tag: 'item', attributes: [{ field: 'name', attr: 'name' }] }],
    } as XmlBinding<unknown>)).toBe('<list><total>5</total><item name="a" /><item name="b" /></list>')
  })

  test('childTags + childRecord', () => {
    const output = {
      count: 2,
      env: { A: '1', B: '2' },
    }
    expect(render('config', output, {
      type: 'tag',
      childTags: [{ field: 'count', tag: 'count' }],
      childRecord: { field: 'env', tag: 'var', keyAttr: 'key' },
    } as XmlBinding<unknown>)).toBe('<config><count>2</count><var key="A">1</var><var key="B">2</var></config>')
  })
})

// =============================================================================
// Self-closing (no body, no nested)
// =============================================================================

describe('buildOutputTree — self-closing', () => {
  test('object with attributes only', () => {
    expect(render('ping', { status: 'ok' }, {
      type: 'tag',
      attributes: [{ field: 'status', attr: 'status' }],
    } as XmlBinding<unknown>)).toBe('<ping status="ok" />')
  })

  test('empty object with no bindings', () => {
    expect(render('noop', {}, TAG)).toBe('<noop />')
  })
})

// =============================================================================
// Error cases
// =============================================================================

describe('buildOutputTree — errors', () => {
  test('throws for non-tag binding with object output', () => {
    expect(() => build('tool', { x: 1 }, { type: 'omit' } as unknown as XmlBinding<unknown>))
      .toThrow('buildOutputTree: tool output <tool> is missing required xmlOutput binding')
  })
})

// =============================================================================
// XPath integration — verify the tree works with DOM + XPath
// =============================================================================

describe('buildOutputTree — XPath integration', () => {
  test('XPath extracts childTag text content', () => {
    const tree = build('read', { content: 'hello <world>', lines: 42 }, {
      type: 'tag',
      childTags: [{ field: 'content', tag: 'content' }, { field: 'lines', tag: 'lines' }],
    })
    const { root } = outputToDOM(tree)
    const result = evaluateXPath('content', root, null, null,
      evaluateXPath.ALL_RESULTS_TYPE,
      { language: evaluateXPath.XQUERY_3_1_LANGUAGE }) as Node[]
    expect(result[0]?.textContent).toBe('hello <world>')
  })

  test('XPath count on items', () => {
    const tree = build('list', ['a', 'b', 'c'], {
      type: 'tag',
      items: { tag: 'item' },
    } as XmlBinding<unknown>)
    const { root } = outputToDOM(tree)
    const result = evaluateXPath('count(item)', root, null, null,
      evaluateXPath.ALL_RESULTS_TYPE,
      { language: evaluateXPath.XQUERY_3_1_LANGUAGE })
    expect(result[0]).toBe(3)
  })

  test('XPath attribute query on items', () => {
    const tree = build('tree', [{ file: 'a.ts', type: 'file' }, { file: 'b/', type: 'dir' }], {
      type: 'tag',
      items: { tag: 'entry', attributes: ['file', 'type'] },
    } as XmlBinding<unknown>)
    const { root } = outputToDOM(tree)
    const result = evaluateXPath('entry[@type="dir"]/@file', root, null, null,
      evaluateXPath.ALL_RESULTS_TYPE,
      { language: evaluateXPath.XQUERY_3_1_LANGUAGE }) as Attr[]
    expect(result[0]?.value).toBe('b/')
  })

  test('XPath . on scalar returns full element', () => {
    const tree = build('add', 7, TAG)
    const { root } = outputToDOM(tree)
    const result = evaluateXPath('.', root, null, null,
      evaluateXPath.ALL_RESULTS_TYPE,
      { language: evaluateXPath.XQUERY_3_1_LANGUAGE }) as Node[]
    expect(result[0]?.textContent).toBe('7')
  })

  test('DOM round-trip preserves structure', () => {
    const tree = build('read', { content: 'const x = <T>()', lines: 42 }, {
      type: 'tag',
      childTags: [{ field: 'content', tag: 'content' }, { field: 'lines', tag: 'lines' }],
    })
    const { root } = outputToDOM(tree)
    const roundTripped = outputFromDOM(root)
    expect(outputToText(roundTripped)).toBe(outputToText(tree))
  })

  test('DOM round-trip with items preserves attributes', () => {
    const tree = build('list', [{ file: 'a.ts', match: 'hello' }], {
      type: 'tag',
      items: { tag: 'item', attributes: ['file'], body: 'match' },
    } as XmlBinding<unknown>)
    const { root } = outputToDOM(tree)
    const roundTripped = outputFromDOM(root)
    expect(outputToText(roundTripped)).toBe('<list><item file="a.ts">hello</item></list>')
  })
})
