
import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import * as fs from 'fs'
import * as path from 'path'
import * as os from 'os'

import {
  queryOutput,
  renderFilteredResult,
  renderResultBlock,
  QueryPatterns,
} from '../mact-output-query'

import {
  renderResult,
  renderResultBody,
  renderVoidResult,
  renderStringResult,
  renderScalarResult,
  renderArrayResult,
  renderObjectResult,
  renderOutField,
  renderShellResult,
  renderReadResult,
  renderWriteResult,
  renderGrepResult,
  isValidResultBlock,
  extractToolName,
  parseResultBlock,
} from '../mact-output-renderer'

import {
  getResultPath,
  persistResult,
  loadResult,
  ensureResultsDir,
  cleanupResults,
} from '../mact-result-persistence'

describe('Mact Output Query', () => {
  describe('queryOutput', () => {
    it('returns full output when no query', () => {
      const output = { stdout: 'hello', stderr: '', exitCode: 0 }
      const result = queryOutput(output, undefined, '/tmp/test.json')
      
      expect(result.filtered).toEqual(output)
      expect(result.isPartial).toBe(false)
      expect(result.fullPath).toBe('/tmp/test.json')
    })

    it('returns full output for root query', () => {
      const output = { stdout: 'hello', stderr: '', exitCode: 0 }
      const result = queryOutput(output, '$', '/tmp/test.json')
      
      expect(result.filtered).toEqual(output)
      expect(result.isPartial).toBe(false)
    })

    it('filters to single field', () => {
      const output = { stdout: 'hello', stderr: '', exitCode: 0 }
      const result = queryOutput(output, '$.stdout', '/tmp/test.json')
      
      expect(result.filtered).toBe('hello')
      expect(result.isPartial).toBe(true)
    })

    it('filters to nested field', () => {
      const output = { 
        items: [{ file: 'a.ts', match: 'TODO' }, { file: 'b.ts', match: 'FIXME' }]
      }
      const result = queryOutput(output, '$.items[0].file', '/tmp/test.json')
      
      expect(result.filtered).toBe('a.ts')
      expect(result.isPartial).toBe(true)
    })

    it('returns array for array queries', () => {
      const output = { items: ['a', 'b', 'c'] }
      const result = queryOutput(output, '$.items[*]', '/tmp/test.json')
      
      expect(result.filtered).toEqual(['a', 'b', 'c'])
      expect(result.isPartial).toBe(true)
    })

    it('returns empty array for non-matching query', () => {
      const output = { stdout: 'hello' }
      const result = queryOutput(output, '$.nonexistent', '/tmp/test.json')
      
      // Non-matching valid query returns empty array
      expect(result.filtered).toEqual([])
      expect(result.isPartial).toBe(true)
    })
  })

  describe('QueryPatterns', () => {
    it('provides common patterns', () => {
      expect(QueryPatterns.full).toBe('$')
      expect(QueryPatterns.first).toBe('$[0]')
      expect(QueryPatterns.field('stdout')).toBe('$.stdout')
      expect(QueryPatterns.nested('items', '0', 'file')).toBe('$.items.0.file')
    })
  })
})

describe('Mact Output Renderer', () => {
  describe('renderVoidResult', () => {
    it('renders void result', () => {
      const result = renderVoidResult('write')
      expect(result).toBe('<|result:write>\n<result|>')
      expect(isValidResultBlock(result)).toBe(true)
    })
  })

  describe('renderStringResult', () => {
    it('renders pure string result', () => {
      const result = renderStringResult('read', 'console.log("hello");')
      expect(result).toBe('<|result:read>\nconsole.log("hello");\n<result|>')
      expect(isValidResultBlock(result)).toBe(true)
    })

    it('handles multi-line content', () => {
      const content = 'line1\nline2\nline3'
      const result = renderStringResult('read', content)
      expect(result).toContain('line1\nline2\nline3')
      expect(extractToolName(result)).toBe('read')
    })
  })

  describe('renderArrayResult', () => {
    it('renders array as JSON', () => {
      const items = [{ file: 'a.ts', match: 'TODO' }]
      const result = renderArrayResult('grep', items)
      
      expect(result).toContain('<|out:items>')
      expect(result).toContain('[{"file":"a.ts","match":"TODO"}]')
      expect(isValidResultBlock(result)).toBe(true)
    })
  })

  describe('renderObjectResult', () => {
    it('renders object with out fields', () => {
      const output = { mode: 'completed', exitCode: 0 }
      const result = renderObjectResult('shell', output)
      
      expect(result).toContain('<|out:mode>completed<out|>')
      expect(result).toContain('<|out:exitCode>0<out|>')
      expect(isValidResultBlock(result)).toBe(true)
    })

    it('handles string fields with raw text', () => {
      const output = { stdout: 'hello\nworld', exitCode: 0 }
      const result = renderObjectResult('shell', output)
      
      expect(result).toContain('<|out:stdout>')
      expect(result).toContain('hello\nworld')
      expect(result).toContain('<out|>')
    })
  })

  describe('renderOutField', () => {
    it('renders short string inline', () => {
      expect(renderOutField('name', 'value')).toBe('<|out:name>value<out|>')
    })

    it('renders long string multi-line', () => {
      const longValue = 'a'.repeat(100)
      const result = renderOutField('content', longValue)
      expect(result).toContain('<|out:content>\n')
      expect(result).toContain('\n<out|>')
    })

    it('renders non-string as JSON', () => {
      expect(renderOutField('count', 42)).toBe('<|out:count>42<out|>')
      expect(renderOutField('flag', true)).toBe('<|out:flag>true<out|>')
      expect(renderOutField('items', [1, 2, 3])).toBe('<|out:items>[1,2,3]<out|>')
    })
  })

  describe('renderShellResult', () => {
    it('renders shell output', () => {
      const result = renderShellResult('hello', '', 0)
      
      expect(result).toContain('<|result:shell>')
      expect(result).toContain('<|out:stdout>hello<out|>')
      expect(result).toContain('<|out:exitCode>0<out|>')
    })

    it('omits empty stderr', () => {
      const result = renderShellResult('hello', '', 0)
      // stderr should be undefined (omitted) when empty
      expect(result).not.toContain('stderr')
    })
  })

  describe('renderReadResult', () => {
    it('renders read as pure string', () => {
      const result = renderReadResult('file content here')
      expect(result).toBe('<|result:read>\nfile content here\n<result|>')
    })
  })

  describe('renderWriteResult', () => {
    it('renders write as void', () => {
      const result = renderWriteResult()
      expect(result).toBe('<|result:write>\n<result|>')
    })
  })

  describe('parseResultBlock', () => {
    it('parses pure string result', () => {
      const block = '<|result:read>\ncontent\n<result|>'
      const parsed = parseResultBlock(block)
      
      expect(parsed).not.toBeNull()
      expect(parsed!.toolName).toBe('read')
      expect(parsed!.content).toBe('content')
    })

    it('parses object result with out fields', () => {
      const block = '<|result:shell>\n<|out:mode>completed<out|>\n<|out:exitCode>0<out|>\n<result|>'
      const parsed = parseResultBlock(block)
      
      expect(parsed).not.toBeNull()
      expect(parsed!.toolName).toBe('shell')
      expect(parsed!.fields.get('mode')).toBe('completed')
      expect(parsed!.fields.get('exitCode')).toBe(0)
    })
  })
})

describe('Mact Result Persistence', () => {
  let tempDir: string
  let originalM: string | undefined

  beforeEach(() => {
    tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'mact-test-'))
    originalM = process.env.M
    process.env.M = tempDir
  })

  afterEach(() => {
    if (originalM !== undefined) {
      process.env.M = originalM
    } else {
      delete process.env.M
    }
    
    // Cleanup temp directory
    if (fs.existsSync(tempDir)) {
      fs.rmSync(tempDir, { recursive: true, force: true })
    }
  })

  describe('persistResult and loadResult', () => {
    it('persists and loads a result', () => {
      const result = { stdout: 'hello', exitCode: 0 }
      const resultPath = persistResult(result, 'turn-1', 'call-1')
      
      expect(fs.existsSync(resultPath)).toBe(true)
      expect(resultPath).toContain('turn-1-call-1.json')
      
      const loaded = loadResult('turn-1', 'call-1')
      expect(loaded).toEqual(result)
    })

    it('throws when loading non-existent result', () => {
      expect(() => loadResult('turn-x', 'call-y')).toThrow('Result not found')
    })
  })

  describe('cleanupResults', () => {
    it('removes old results', () => {
      const result = { data: 'test' }
      persistResult(result, 'turn-1', 'call-1')
      
      // Should not delete recent results
      const deleted = cleanupResults(1000) // 1 second max age
      expect(deleted).toBe(0)
      
      // Modify file to be old
      const resultPath = getResultPath('turn-1', 'call-1')
      const oldTime = new Date(Date.now() - 5000) // 5 seconds ago
      fs.utimesSync(resultPath, oldTime, oldTime)
      
      // Now should delete
      const deletedOld = cleanupResults(1000)
      expect(deletedOld).toBe(1)
      expect(fs.existsSync(resultPath)).toBe(false)
    })
  })
})
