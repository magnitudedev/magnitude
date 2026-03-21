import { describe, test, expect } from 'bun:test'
import { Schema } from '@effect/schema'
import { generateXmlToolDoc, generateXmlToolGroupDoc, type XmlToolDocEntry } from './xml-docs'
import { ToolImageSchema } from '@magnitudedev/tools'
import type { XmlTagBinding } from './types'
import type { XmlOutputBinding } from './xml-binding'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeTool<
  I extends Schema.Schema.AnyNoContext,
  O extends Schema.Schema.AnyNoContext = typeof Schema.Void,
>(config: {
  name: string
  group?: string
  description?: string
  inputSchema: I
  outputSchema?: O
  argMapping?: readonly string[]
  xmlInput?: XmlTagBinding
  xmlOutput?: XmlOutputBinding<unknown>
  bindings?: {
    xmlInput: ({ readonly type: 'tag' } & XmlTagBinding)
    xmlOutput?: { readonly type: 'tag'; readonly [k: string]: unknown }
  }
}): XmlToolDocEntry | null {
  const xmlInputWithType = config.bindings?.xmlInput
  const xmlInput = config.xmlInput ?? (xmlInputWithType
    ? (() => {
        const { type: _type, ...binding } = xmlInputWithType
        return binding
      })()
    : undefined)
  if (!xmlInput) return null

  const xmlOutputWithType = config.bindings?.xmlOutput
  const xmlOutput = config.xmlOutput ?? (xmlOutputWithType
    ? (() => {
        const { type: _type, ...binding } = xmlOutputWithType
        return binding as XmlOutputBinding<unknown>
      })()
    : undefined)

  return {
    name: config.name,
    group: config.group,
    description: config.description ?? '',
    inputSchema: config.inputSchema,
    outputSchema: (config.outputSchema ?? Schema.Void) as O,
    xmlInput,
    xmlOutput,
  }
}

function expectTool(tool: XmlToolDocEntry | null): XmlToolDocEntry {
  if (!tool) throw new Error('expected xml-bound test tool')
  return tool
}

// ---------------------------------------------------------------------------
// No binding / omit binding
// ---------------------------------------------------------------------------

describe('no binding or omit', () => {
  test('tool with no bindings returns null', () => {
    const tool = makeTool({
      name: 'tree',
      inputSchema: Schema.Struct({ path: Schema.String }),
    })
    expect(tool).toBeNull()
  })


})

// ---------------------------------------------------------------------------
// Self-closing tags (attributes only)
// ---------------------------------------------------------------------------

describe('self-closing tags', () => {
  test('single attribute with description', () => {
    const tool = makeTool({
      name: 'read',
      group: 'fs',
      description: 'Read file content',
      inputSchema: Schema.Struct({
        path: Schema.String.annotations({ description: 'Relative path from cwd' }),
      }),
      argMapping: ['path'],
      bindings: { xmlInput: { type: 'tag', attributes: [{ field: 'path', attr: 'path' }], selfClosing: true } },
    })
    const doc = generateXmlToolDoc(expectTool(tool))
    expect(doc).toContain('Read file content')
    expect(doc).toContain('<fs-read')
    expect(doc).toContain('path="..."')
    expect(doc).toContain('Relative path from cwd')
    expect(doc).toContain('/>')
  })

  test('multiple attributes', () => {
    const tool = makeTool({
      name: 'search',
      description: 'Search files',
      inputSchema: Schema.Struct({
        pattern: Schema.String.annotations({ description: 'Regex pattern' }),
        path: Schema.String.annotations({ description: 'Directory' }),
      }),
      argMapping: ['pattern', 'path'],
      bindings: { xmlInput: { type: 'tag', attributes: [{ field: 'pattern', attr: 'pattern' }, { field: 'path', attr: 'path' }], selfClosing: true } },
    })
    const doc = generateXmlToolDoc(expectTool(tool))
    expect(doc).toContain('<search')
    expect(doc).toContain('pattern="..."')
    expect(doc).toContain('path="..."')
    expect(doc).toContain('Regex pattern')
    expect(doc).toContain('Directory')
  })
})

// ---------------------------------------------------------------------------
// Tags with body
// ---------------------------------------------------------------------------

describe('body tags', () => {
  test('body-only tag', () => {
    const tool = makeTool({
      name: 'shell',
      description: 'Execute a shell command',
      inputSchema: Schema.Struct({ command: Schema.String }),
      argMapping: ['command'],
      bindings: { xmlInput: { type: 'tag', body: 'command' } },
    })
    const doc = generateXmlToolDoc(expectTool(tool))
    expect(doc).toContain('<shell>command</shell>')
  })

  test('attributes + body', () => {
    const tool = makeTool({
      name: 'write',
      description: 'Write content to file',
      inputSchema: Schema.Struct({
        path: Schema.String.annotations({ description: 'File path' }),
        content: Schema.String.annotations({ description: 'Content to write' }),
      }),
      argMapping: ['path', 'content'],
      bindings: { xmlInput: { type: 'tag', attributes: [{ field: 'path', attr: 'path' }], body: 'content' } },
    })
    const doc = generateXmlToolDoc(expectTool(tool))
    expect(doc).toContain('path="..."')
    expect(doc).toContain('>content</write>')
    expect(doc).toContain('File path')
    expect(doc).toContain('Content to write')
  })
})

// ---------------------------------------------------------------------------
// Tags with children (repeated array elements)
// ---------------------------------------------------------------------------

describe('children tags', () => {
  test('children with attributes and body', () => {
    const EditSchema = Schema.Struct({
      from: Schema.String.annotations({ description: 'Start anchor' }),
      to: Schema.optional(Schema.String.annotations({ description: 'End anchor' })),
      content: Schema.optional(Schema.String.annotations({ description: 'New content' })),
    })
    const tool = makeTool({
      name: 'edit',
      description: 'Edit a file',
      inputSchema: Schema.Struct({
        path: Schema.String.annotations({ description: 'File path' }),
        edits: Schema.Array(EditSchema),
      }),
      argMapping: ['path', 'edits'],
      bindings: {
        xmlInput: {
          type: 'tag',
          attributes: [{ field: 'path', attr: 'path' }],
          children: [{
            field: 'edits',
            tag: 'edit-item',
            attributes: [{ field: 'from', attr: 'from' }, { field: 'to', attr: 'to' }],
            body: 'content',
          }],
        },
      },
    })
    const doc = generateXmlToolDoc(expectTool(tool))
    expect(doc).toContain('Edit a file')
    expect(doc).toContain('path="..."')
    expect(doc).toContain('File path')
    expect(doc).toContain('from="..."')
    expect(doc).toContain('Start anchor')
    expect(doc).toContain('to="..."')
    expect(doc).toContain('optional')
    expect(doc).toContain('End anchor')
    expect(doc).toContain('>content</edit-item>')
    expect(doc).toContain('<!-- ...more edit-items -->')
  })
})

// ---------------------------------------------------------------------------
// Tag name override
// ---------------------------------------------------------------------------

describe('tag name override', () => {
  test('custom tag name via binding', () => {
    const tool = makeTool({
      name: 'fileRead',
      description: 'Read a file',
      inputSchema: Schema.Struct({ path: Schema.String }),
      argMapping: ['path'],
      bindings: { xmlInput: { type: 'tag', tag: 'read', attributes: [{ field: 'path', attr: 'path' }], selfClosing: true } },
    })
    const doc = generateXmlToolDoc(expectTool(tool))
    expect(doc).toContain('<read')
    expect(doc).not.toContain('fileRead')
  })
})

// ---------------------------------------------------------------------------
// Group documentation
// ---------------------------------------------------------------------------

describe('group documentation', () => {
  const readTool = makeTool({
    name: 'read',
    group: 'fs',
    description: 'Read file',
    inputSchema: Schema.Struct({ path: Schema.String }),
    argMapping: ['path'],
    bindings: { xmlInput: { type: 'tag', attributes: [{ field: 'path', attr: 'path' }], selfClosing: true } },
  })

  const writeTool = makeTool({
    name: 'write',
    group: 'fs',
    description: 'Write file',
    inputSchema: Schema.Struct({
      path: Schema.String,
      content: Schema.String,
    }),
    argMapping: ['path', 'content'],
    bindings: { xmlInput: { type: 'tag', attributes: [{ field: 'path', attr: 'path' }], body: 'content' } },
  })

  const noBindingTool = makeTool({
    name: 'tree',
    group: 'fs',
    description: 'List directory',
    inputSchema: Schema.Struct({ path: Schema.String }),
  })

  test('generates group header with tool docs', () => {
    if (!readTool || !writeTool) throw new Error('expected tools with xml bindings')
    const doc = generateXmlToolGroupDoc('fs', [readTool, writeTool])
    expect(doc).toContain('### fs')
    expect(doc).toContain('Read file')
    expect(doc).toContain('Write file')
  })

  test('skips tools with no XML binding', () => {
    if (!readTool) throw new Error('expected readTool with xml binding')
    const tools = [readTool, noBindingTool].filter((tool): tool is XmlToolDocEntry => tool !== null)
    const doc = generateXmlToolGroupDoc('fs', tools)
    expect(doc).toContain('Read file')
    expect(doc).not.toContain('List directory')
  })

  test('filters implicit tools via defKey lookup', () => {
    if (!readTool || !writeTool) throw new Error('expected tools with xml bindings')
    const defKeyLookup = new Map<XmlToolDocEntry, string>([
      [readTool, 'fileRead'],
      [writeTool, 'fileWrite'],
    ])
    const doc = generateXmlToolGroupDoc('fs', [readTool, writeTool], ['fileRead'], defKeyLookup)
    expect(doc).not.toContain('Read file')
    expect(doc).toContain('Write file')
  })

  test('returns empty string when all tools filtered', () => {
    const tools = [noBindingTool].filter((tool): tool is XmlToolDocEntry => tool !== null)
    const doc = generateXmlToolGroupDoc('fs', tools)
    expect(doc).toBe('')
  })
})

// ---------------------------------------------------------------------------
// ChildTags documentation
// ---------------------------------------------------------------------------

describe('childTags', () => {
  test('childTags with descriptions from nested object schema', () => {
    const tool = makeTool({
      name: 'create',
      group: 'agent',
      description: 'Create a new agent',
      inputSchema: Schema.Struct({
        id: Schema.String.annotations({ description: 'Agent ID' }),
        options: Schema.Struct({
          type: Schema.String.annotations({ description: 'Agent type' }),
          goal: Schema.String.annotations({ description: 'Short summary of what to accomplish' }),
          prompt: Schema.String.annotations({ description: 'Detailed instructions' }),
        }),
      }),
      argMapping: ['id', 'options'],
      bindings: {
        xmlInput: {
          type: 'tag',
          attributes: [{ field: 'id', attr: 'id' }],
          childTags: [
            { field: 'options.type', tag: 'type' },
            { field: 'options.goal', tag: 'goal' },
            { field: 'options.prompt', tag: 'prompt' },
          ],
        },
      },
    })
    const doc = generateXmlToolDoc(expectTool(tool))
    expect(doc).toContain('Create a new agent')
    expect(doc).toContain('<agent-create')
    expect(doc).toContain('<type>type</type>')
    expect(doc).toContain('Agent type')
    expect(doc).toContain('<goal>goal</goal>')
    expect(doc).toContain('<prompt>prompt</prompt>')
    expect(doc).toContain('</agent-create>')
  })

  test('childTags with optional nested field', () => {
    const tool = makeTool({
      name: 'config',
      inputSchema: Schema.Struct({
        opts: Schema.Struct({
          host: Schema.String.annotations({ description: 'Server host' }),
          port: Schema.optional(Schema.Number.annotations({ description: 'Server port' })),
        }),
      }),
      argMapping: ['opts'],
      bindings: {
        xmlInput: {
          type: 'tag',
          childTags: [
            { field: 'opts.host', tag: 'host' },
            { field: 'opts.port', tag: 'port' },
          ],
        },
      },
    })
    const doc = generateXmlToolDoc(expectTool(tool))
    expect(doc).toContain('<host>host</host>')
    expect(doc).toContain('Server host')
    expect(doc).toContain('<port>port</port>')
    expect(doc).toContain('optional')
    expect(doc).toContain('number')
    expect(doc).toContain('Server port')
  })
})

// ---------------------------------------------------------------------------
// ChildRecord documentation
// ---------------------------------------------------------------------------

describe('childRecord', () => {
  test('childRecord with description from record field', () => {
    const tool = makeTool({
      name: 'propose',
      description: 'Propose a plan',
      inputSchema: Schema.Struct({
        title: Schema.String.annotations({ description: 'Plan title' }),
        description: Schema.String.annotations({ description: 'Plan description' }),
        criteria: Schema.optional(Schema.Record({ key: Schema.String, value: Schema.String })).annotations({
          description: 'Criterion ID -> description map',
        }),
      }),
      argMapping: ['title', 'description', 'criteria'],
      bindings: {
        xmlInput: {
          type: 'tag',
          attributes: [{ field: 'title', attr: 'title' }, { field: 'description', attr: 'description' }],
          childRecord: { field: 'criteria', tag: 'criterion', keyAttr: 'id' },
        },
      },
    })
    const doc = generateXmlToolDoc(expectTool(tool))
    expect(doc).toContain('Propose a plan')
    expect(doc).toContain('title="..."')
    expect(doc).toContain('description="..."')
    expect(doc).toContain('<criterion id="...">value</criterion>')
    expect(doc).toContain('<!-- ...more')
    expect(doc).toContain('</propose>')
  })
})

// ---------------------------------------------------------------------------
// Output documentation
// ---------------------------------------------------------------------------

describe('output documentation', () => {
  test('void output omits returns section', () => {
    const tool = makeTool({
      name: 'write',
      inputSchema: Schema.Struct({ path: Schema.String, content: Schema.String }),
      outputSchema: Schema.Void,
      bindings: { xmlInput: { type: 'tag', attributes: [{ field: 'path', attr: 'path' }], body: 'content' } },
    })
    const doc = generateXmlToolDoc(expectTool(tool))
    expect(doc).not.toContain('Returns')
  })

  test('string output shows Returns: string', () => {
    const tool = makeTool({
      name: 'read',
      group: 'fs',
      inputSchema: Schema.Struct({ path: Schema.String }),
      outputSchema: Schema.String,
      bindings: { xmlInput: { type: 'tag', attributes: [{ field: 'path', attr: 'path' }], selfClosing: true } },
    })
    const doc = generateXmlToolDoc(expectTool(tool))
    expect(doc).toContain('Returns: string')
    expect(doc).toContain('<fs-read>...</fs-read>')
  })

  test('image output shows Returns: image', () => {
    const tool = makeTool({
      name: 'screenshot',
      group: 'browser',
      inputSchema: Schema.Struct({}),
      outputSchema: ToolImageSchema,
      bindings: { xmlInput: { type: 'tag' } },
    })
    const doc = generateXmlToolDoc(expectTool(tool))
    expect(doc).toContain('Returns: image')
    expect(doc).toContain('<browser-screenshot>[image]</browser-screenshot>')
  })

  test('struct output shows child tags', () => {
    const tool = makeTool({
      name: 'shell',
      inputSchema: Schema.Struct({ command: Schema.String }),
      outputSchema: Schema.Struct({
        stdout: Schema.String,
        stderr: Schema.String,
        exitCode: Schema.Number,
      }),
      bindings: { xmlInput: { type: 'tag', body: 'command' } },
    })
    const doc = generateXmlToolDoc(expectTool(tool))
    expect(doc).toContain('Returns:')
    expect(doc).toContain('<stdout>stdout</stdout>')
    expect(doc).toContain('<stderr>stderr</stderr>')
    expect(doc).toContain('<exitCode>exitCode</exitCode>')
    expect(doc).toContain('number')
  })

  test('struct output with image field shows image placeholder', () => {
    const tool = makeTool({
      name: 'inspect',
      inputSchema: Schema.Struct({}),
      outputSchema: Schema.Struct({
        screenshot: ToolImageSchema,
        title: Schema.String,
      }),
      bindings: { xmlInput: { type: 'tag' } },
    })
    const doc = generateXmlToolDoc(expectTool(tool))
    expect(doc).toContain('<screenshot>[image]</screenshot>')
    expect(doc).toContain('<title>title</title>')
  })

  test('array-struct output shows item with attrs', () => {
    const tool = makeTool({
      name: 'tree',
      group: 'fs',
      inputSchema: Schema.Struct({ path: Schema.String }),
      outputSchema: Schema.Array(Schema.Struct({
        path: Schema.String,
        name: Schema.String,
        type: Schema.Literal('file', 'dir'),
      })),
      bindings: { xmlInput: { type: 'tag', attributes: [{ field: 'path', attr: 'path' }], selfClosing: true } },
    })
    const doc = generateXmlToolDoc(expectTool(tool))
    expect(doc).toContain('Returns:')
    expect(doc).toContain('<item')
    expect(doc).toContain('path="..."')
    expect(doc).toContain('<!-- ...more items')
  })

  test('explicit output binding overrides defaults', () => {
    const tool = makeTool({
      name: 'shell',
      inputSchema: Schema.Struct({ command: Schema.String }),
      outputSchema: Schema.Struct({
        stdout: Schema.String,
        stderr: Schema.String,
        exitCode: Schema.Number,
      }),
      bindings: {
        xmlInput: { type: 'tag', body: 'command' },
        xmlOutput: {
          type: 'tag',
          attributes: [{ field: 'exitCode', attr: 'exitCode' }],
          childTags: [
            { field: 'stdout', tag: 'stdout' },
            { field: 'stderr', tag: 'stderr' },
          ],
        },
      },
    })
    const doc = generateXmlToolDoc(expectTool(tool))
    expect(doc).toContain('Returns:')
    expect(doc).toContain('exitCode="..."')
    expect(doc).toContain('<stdout>stdout</stdout>')
    expect(doc).toContain('<stderr>stderr</stderr>')
  })

  test('explicit output binding with body still works', () => {
    const tool = makeTool({
      name: 'read',
      group: 'fs',
      inputSchema: Schema.Struct({ path: Schema.String }),
      outputSchema: Schema.Struct({ content: Schema.String }),
      bindings: {
        xmlInput: { type: 'tag', attributes: [{ field: 'path', attr: 'path' }], selfClosing: true },
        xmlOutput: { type: 'tag', body: 'content' },
      },
    })
    const doc = generateXmlToolDoc(expectTool(tool))
    expect(doc).toContain('Returns:')
    expect(doc).toContain('<fs-read>content</fs-read>')
  })

  test('explicit output childTags render image placeholder for image fields', () => {
    const tool = makeTool({
      name: 'inspect',
      inputSchema: Schema.Struct({}),
      outputSchema: Schema.Struct({
        screenshot: ToolImageSchema,
        title: Schema.String,
      }),
      bindings: {
        xmlInput: { type: 'tag' },
        xmlOutput: {
          type: 'tag',
          childTags: [
            { field: 'screenshot', tag: 'screenshot' },
            { field: 'title', tag: 'title' },
          ],
        },
      },
    })
    const doc = generateXmlToolDoc(expectTool(tool))
    expect(doc).toContain('<screenshot>[image]</screenshot>')
    expect(doc).toContain('<title>title</title>')
  })

  test('explicit output items binding renders item attrs without body', () => {
    const tool = makeTool({
      name: 'tree',
      group: 'fs',
      inputSchema: Schema.Struct({ path: Schema.String }),
      outputSchema: Schema.Array(Schema.Struct({
        path: Schema.String,
        name: Schema.String,
        type: Schema.Literal('file', 'dir'),
        depth: Schema.Number,
      })),
      bindings: {
        xmlInput: { type: 'tag', attributes: [{ field: 'path', attr: 'path' }], selfClosing: true },
        xmlOutput: {
          type: 'tag',
          items: {
            tag: 'entry',
            attributes: [
              { attr: 'path', field: 'path' },
              { attr: 'name', field: 'name' },
              { attr: 'type', field: 'type' },
              { attr: 'depth', field: 'depth' },
            ],
          },
        },
      },
    })
    const doc = generateXmlToolDoc(expectTool(tool))
    expect(doc).toContain('Returns:')
    expect(doc).toContain('<fs-tree>')
    expect(doc).toContain('<entry path="..." name="..." type="..." depth="..." />')
    expect(doc).toContain('</fs-tree>')
  })

  test('explicit output items binding renders item attrs + body', () => {
    const tool = makeTool({
      name: 'search',
      group: 'fs',
      inputSchema: Schema.Struct({ pattern: Schema.String }),
      outputSchema: Schema.Array(Schema.Struct({
        file: Schema.String,
        match: Schema.String,
      })),
      bindings: {
        xmlInput: { type: 'tag', attributes: [{ field: 'pattern', attr: 'pattern' }], selfClosing: true },
        xmlOutput: { type: 'tag', items: { tag: 'item', attributes: [{ attr: 'file', field: 'file' }], body: 'match' } },
      },
    })
    const doc = generateXmlToolDoc(expectTool(tool))
    expect(doc).toContain('Returns:')
    expect(doc).toContain('<fs-search>')
    expect(doc).toContain('<item file="...">a string</item>')
    expect(doc).toContain('</fs-search>')
  })
})

// ---------------------------------------------------------------------------
// Type annotations
// ---------------------------------------------------------------------------

describe('type annotations', () => {
  test('number fields get number annotation', () => {
    const tool = makeTool({
      name: 'test',
      inputSchema: Schema.Struct({
        count: Schema.Number.annotations({ description: 'Item count' }),
      }),
      bindings: { xmlInput: { type: 'tag', attributes: [{ field: 'count', attr: 'count' }], selfClosing: true } },
    })
    const doc = generateXmlToolDoc(expectTool(tool))
    expect(doc).toContain('number')
    expect(doc).toContain('Item count')
  })

  test('boolean fields get boolean annotation', () => {
    const tool = makeTool({
      name: 'test',
      inputSchema: Schema.Struct({
        verbose: Schema.Boolean,
      }),
      bindings: { xmlInput: { type: 'tag', attributes: [{ field: 'verbose', attr: 'verbose' }], selfClosing: true } },
    })
    const doc = generateXmlToolDoc(expectTool(tool))
    expect(doc).toContain('boolean')
  })

  test('literal union gets variant list', () => {
    const tool = makeTool({
      name: 'test',
      inputSchema: Schema.Struct({
        mode: Schema.Literal('fast', 'slow'),
      }),
      bindings: { xmlInput: { type: 'tag', attributes: [{ field: 'mode', attr: 'mode' }], selfClosing: true } },
    })
    const doc = generateXmlToolDoc(expectTool(tool))
    expect(doc).toContain('"fast" | "slow"')
  })

  test('optional fields annotated', () => {
    const tool = makeTool({
      name: 'test',
      inputSchema: Schema.Struct({
        path: Schema.String,
        glob: Schema.optional(Schema.String.annotations({ description: 'Glob filter' })),
      }),
      bindings: { xmlInput: { type: 'tag', attributes: [{ field: 'path', attr: 'path' }, { field: 'glob', attr: 'glob' }], selfClosing: true } },
    })
    const doc = generateXmlToolDoc(expectTool(tool))
    expect(doc).toContain('optional')
    expect(doc).toContain('Glob filter')
  })
})