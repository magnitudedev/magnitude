import { describe, test, expect } from 'bun:test'
import { Effect } from 'effect'
import { createAgentTestHarness } from '../harness'
import { MockTurnScriptTag } from '../turn-script'
import {
  createVirtualFs,
  createFsReadHandler,
  createFsWriteHandler,
  createEditHandler,
  createFsTreeHandler,
  createFsSearchHandler,
} from '../virtual-fs'

describe('virtual fs handlers', () => {
  test('read returns seeded content', () => {
    const files = createVirtualFs({ 'src/index.ts': 'console.log("hi")' })
    const read = createFsReadHandler(files)
    expect(read({ path: 'src/index.ts' })).toBe('console.log("hi")')
  })

  test('read with offset/limit', () => {
    const files = createVirtualFs({
      'a.txt': ['line1', 'line2', 'line3', 'line4', 'line5'].join('\n'),
    })
    const read = createFsReadHandler(files)
    expect(read({ path: 'a.txt', offset: 2, limit: 3 })).toBe(
      'line2\nline3\nline4\n... (1 more lines remaining. Use offset=5 to continue reading.)',
    )
  })

  test('read missing file', () => {
    const read = createFsReadHandler(createVirtualFs())
    expect(() => read({ path: 'missing.txt' })).toThrow('Failed to read missing.txt')
  })

  test('write creates file', () => {
    const files = createVirtualFs()
    const write = createFsWriteHandler(files)
    write({ path: 'output.txt', content: 'content' })
    expect(files.get('output.txt')).toBe('content')
  })

  test('write overwrites', () => {
    const files = createVirtualFs({ 'output.txt': 'old' })
    const write = createFsWriteHandler(files)
    write({ path: 'output.txt', content: 'new' })
    expect(files.get('output.txt')).toBe('new')
  })

  test('edit find/replace', () => {
    const files = createVirtualFs({ 'file.txt': 'hello world' })
    const edit = createEditHandler(files)
    edit({ path: 'file.txt', oldString: 'world', newString: 'there' })
    expect(files.get('file.txt')).toBe('hello there')
  })

  test('edit replaceAll', () => {
    const files = createVirtualFs({ 'file.txt': 'a b a b a' })
    const edit = createEditHandler(files)
    const message = edit({ path: 'file.txt', oldString: 'a', newString: 'x', replaceAll: true })
    expect(files.get('file.txt')).toBe('x b x b x')
    expect(message).toBe('Replaced 3 occurrences in file.txt')
  })

  test('edit missing file', () => {
    const edit = createEditHandler(createVirtualFs())
    expect(() => edit({ path: 'missing.txt', oldString: 'a', newString: 'b' })).toThrow(
      'Failed to read missing.txt',
    )
  })

  test('edit old string not found', () => {
    const files = createVirtualFs({ 'file.txt': 'hello' })
    const edit = createEditHandler(files)
    expect(() => edit({ path: 'file.txt', oldString: 'nope', newString: 'x' })).toThrow(
      '<old> content not found in file. Ensure it matches the file exactly.',
    )
  })

  test('tree lists files', () => {
    const files = createVirtualFs({
      'src/index.ts': 'a',
      'src/lib/util.ts': 'b',
      'README.md': 'c',
    })
    const tree = createFsTreeHandler(files)
    expect(tree({ path: '' })).toEqual([
      { path: 'README.md', name: 'README.md', type: 'file', depth: 1 },
      { path: 'src', name: 'src', type: 'dir', depth: 1 },
      { path: 'src/index.ts', name: 'index.ts', type: 'file', depth: 2 },
      { path: 'src/lib', name: 'lib', type: 'dir', depth: 2 },
      { path: 'src/lib/util.ts', name: 'util.ts', type: 'file', depth: 3 },
    ])
  })

  test('grep matches regex', () => {
    const files = createVirtualFs({
      'src/a.ts': 'alpha\nbeta',
      'src/b.ts': 'gamma\nalphabet',
    })
    const search = createFsSearchHandler(files)
    expect(search({ pattern: 'alpha' })).toEqual([
      { file: 'src/a.ts', match: '1|alpha' },
      { file: 'src/b.ts', match: '2|alphabet' },
    ])
  })

  test('grep respects path prefix', () => {
    const files = createVirtualFs({
      'src/a.ts': 'todo here',
      'lib/b.ts': 'todo there',
    })
    const search = createFsSearchHandler(files)
    expect(search({ pattern: 'todo', path: 'src' })).toEqual([{ file: 'src/a.ts', match: '1|todo here' }])
  })

  test('grep respects limit', () => {
    const files = createVirtualFs({
      'a.txt': 'hit\nhit\nhit',
      'b.txt': 'hit\nhit',
    })
    const search = createFsSearchHandler(files)
    expect(search({ pattern: 'hit', limit: 2 })).toEqual([
      { file: 'a.txt', match: '1|hit' },
      { file: 'a.txt', match: '2|hit' },
    ])
  })
})

describe('virtual fs integration with harness', () => {
  test('Agent reads seeded file', async () => {
    const harness = await createAgentTestHarness({
      files: { 'src/index.ts': 'export const x = 1' },
    })

    try {
      await harness.runEffect(
        Effect.flatMap(MockTurnScriptTag, (script) =>
          script.enqueue(
            {
              xml: '<actions><read path="src/index.ts"/></actions><yield/>',
            },
            null,
          ),
        ),
      )

      await harness.user('read file')
      const completed = await harness.wait.event(
        'turn_completed',
        (e) => e.toolCalls.length > 0,
      )

      expect(completed.result.success).toBe(true)
      const readCall = completed.toolCalls.find(
        (c) => c.toolKey === 'fileRead' || (c.group === 'fs' && c.toolName === 'read'),
      )
      expect(readCall?.result.status).toBe('success')
      if (readCall?.result.status === 'success') {
        expect(readCall.result.output).toBe('export const x = 1')
      }
    } finally {
      await harness.dispose()
    }
  })

  test('Agent writes file visible in h.files', async () => {
    const harness = await createAgentTestHarness()

    try {
      await harness.runEffect(
        Effect.flatMap(MockTurnScriptTag, (script) =>
          script.enqueue(
            {
              xml: '<actions><write path="output.txt">content</' + 'write></actions><yield/>',
            },
            null,
          ),
        ),
      )

      await harness.user('write file')
      const completed = await harness.wait.event(
        'turn_completed',
        (e) => e.toolCalls.length > 0,
      )

      expect(completed.result.success).toBe(true)
      const writeCall = completed.toolCalls.find(
        (c) => c.toolKey === 'fileWrite' || (c.group === 'fs' && c.toolName === 'write'),
      )
      expect(writeCall?.result.status).toBe('success')
      expect(harness.files.get('output.txt')).toBe('content')
    } finally {
      await harness.dispose()
    }
  })
})