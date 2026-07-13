import { describe, expect, test } from 'bun:test'
import { expandScratchpadPath } from './expand'

const SP = '/tmp/test-scratchpad'

describe('expandScratchpadPath', () => {
  // --- Basic $M expansion ---

  test('$M bare → scratchpadPath', () => {
    expect(expandScratchpadPath('$M', SP)).toEqual({ path: SP, expanded: true, displayPath: '' })
  })

  test('${M} bare → scratchpadPath', () => {
    expect(expandScratchpadPath('${M}', SP)).toEqual({ path: SP, expanded: true, displayPath: '' })
  })

  test('$M/foo → scratchpadPath/foo', () => {
    expect(expandScratchpadPath('$M/reports/foo.md', SP)).toEqual({
      path: '/tmp/test-scratchpad/reports/foo.md',
      expanded: true,
      displayPath: 'reports/foo.md',
    })
  })

  test('${M}/foo → scratchpadPath/foo', () => {
    expect(expandScratchpadPath('${M}/reports/foo.md', SP)).toEqual({
      path: '/tmp/test-scratchpad/reports/foo.md',
      expanded: true,
      displayPath: 'reports/foo.md',
    })
  })

  // --- Dot-segment stripping before $M ---

  test('./$M/foo → scratchpadPath/foo', () => {
    expect(expandScratchpadPath('./$M/reports/foo.md', SP)).toEqual({
      path: '/tmp/test-scratchpad/reports/foo.md',
      expanded: true,
      displayPath: 'reports/foo.md',
    })
  })

  test('./${M}/foo → scratchpadPath/foo', () => {
    expect(expandScratchpadPath('./${M}/reports/foo.md', SP)).toEqual({
      path: '/tmp/test-scratchpad/reports/foo.md',
      expanded: true,
      displayPath: 'reports/foo.md',
    })
  })

  test('./$M bare → scratchpadPath', () => {
    expect(expandScratchpadPath('./$M', SP)).toEqual({ path: SP, expanded: true, displayPath: '' })
  })

  test('../$M/foo → scratchpadPath/foo', () => {
    expect(expandScratchpadPath('../$M/reports/foo.md', SP)).toEqual({
      path: '/tmp/test-scratchpad/reports/foo.md',
      expanded: true,
      displayPath: 'reports/foo.md',
    })
  })

  test('../$M bare → scratchpadPath', () => {
    expect(expandScratchpadPath('../$M', SP)).toEqual({ path: SP, expanded: true, displayPath: '' })
  })

  test('./../$M/foo → scratchpadPath/foo', () => {
    expect(expandScratchpadPath('./../$M/reports/foo.md', SP)).toEqual({
      path: '/tmp/test-scratchpad/reports/foo.md',
      expanded: true,
      displayPath: 'reports/foo.md',
    })
  })

  test('../${M}/foo → scratchpadPath/foo', () => {
    expect(expandScratchpadPath('../${M}/reports/foo.md', SP)).toEqual({
      path: '/tmp/test-scratchpad/reports/foo.md',
      expanded: true,
      displayPath: 'reports/foo.md',
    })
  })

  // --- Non-$M paths pass through unchanged ---

  test('non-$M path passes through unchanged', () => {
    expect(expandScratchpadPath('src/index.ts', SP)).toEqual({
      path: 'src/index.ts',
      expanded: false,
      displayPath: 'src/index.ts',
    })
  })

  test('./src/foo.ts passes through unchanged', () => {
    expect(expandScratchpadPath('./src/foo.ts', SP)).toEqual({
      path: './src/foo.ts',
      expanded: false,
      displayPath: './src/foo.ts',
    })
  })

  test('../src/foo.ts passes through unchanged', () => {
    expect(expandScratchpadPath('../src/foo.ts', SP)).toEqual({
      path: '../src/foo.ts',
      expanded: false,
      displayPath: '../src/foo.ts',
    })
  })

  test('absolute path passes through unchanged', () => {
    expect(expandScratchpadPath('/etc/passwd', SP)).toEqual({
      path: '/etc/passwd',
      expanded: false,
      displayPath: '/etc/passwd',
    })
  })

  // --- Path traversal protection (boundary check on resolved path) ---

  test('$M/../foo returns original input (traversal escape)', () => {
    const r = expandScratchpadPath('$M/../foo.md', SP)
    expect(r.expanded).toBe(false)
    expect(r.path).toBe('$M/../foo.md')
    expect(r.displayPath).toBe('$M/../foo.md')
  })

  test('$M/../../etc/passwd returns original input', () => {
    const r = expandScratchpadPath('$M/../../etc/passwd', SP)
    expect(r.expanded).toBe(false)
    expect(r.path).toBe('$M/../../etc/passwd')
    expect(r.displayPath).toBe('$M/../../etc/passwd')
  })

  test('./$M/../etc/passwd returns original input', () => {
    const r = expandScratchpadPath('./$M/../etc/passwd', SP)
    expect(r.expanded).toBe(false)
    expect(r.path).toBe('./$M/../etc/passwd')
    expect(r.displayPath).toBe('./$M/../etc/passwd')
  })

  // --- Double-slash escape (the bug) ---

  test('$M//etc/passwd returns original input (escape blocked)', () => {
    const r = expandScratchpadPath('$M//etc/passwd', SP)
    expect(r.expanded).toBe(false)
    expect(r.path).toBe('$M//etc/passwd')
    expect(r.displayPath).toBe('$M//etc/passwd')
  })

  test('$M//reports/foo.md returns original input (escape blocked)', () => {
    const r = expandScratchpadPath('$M//reports/foo.md', SP)
    expect(r.expanded).toBe(false)
    expect(r.path).toBe('$M//reports/foo.md')
    expect(r.displayPath).toBe('$M//reports/foo.md')
  })

  // --- Normalization within scratchpad ---

  test('$M/a/../b normalizes safely', () => {
    expect(expandScratchpadPath('$M/a/../b', SP)).toEqual({
      path: '/tmp/test-scratchpad/b',
      expanded: true,
      displayPath: 'b',
    })
  })

  test('$M/a/./b normalizes safely', () => {
    expect(expandScratchpadPath('$M/a/./b', SP)).toEqual({
      path: '/tmp/test-scratchpad/a/b',
      expanded: true,
      displayPath: 'a/b',
    })
  })

  test('$M/. resolves to scratchpadPath with empty displayPath', () => {
    expect(expandScratchpadPath('$M/.', SP)).toEqual({
      path: '/tmp/test-scratchpad',
      expanded: true,
      displayPath: '',
    })
  })

  test('$M/foo/ trailing slash normalizes', () => {
    expect(expandScratchpadPath('$M/foo/', SP)).toEqual({
      path: '/tmp/test-scratchpad/foo',
      expanded: true,
      displayPath: 'foo',
    })
  })

  // --- Edge cases ---

  test('empty string passes through', () => {
    expect(expandScratchpadPath('', SP)).toEqual({ path: '', expanded: false, displayPath: '' })
  })
})
