import { describe, test, expect } from 'bun:test'
import { mkdtempSync } from 'fs'
import { rm } from 'fs/promises'
import { join } from 'path'
import { tmpdir } from 'os'
import { createStorageClient } from '@magnitudedev/storage'
import type { MagnitudeSlot } from '../../model-slots'
import {
  MEMORY_RELATIVE_PATH,
  MEMORY_TEMPLATE,
  ensureMemoryFile,
  readMemory,
  parseMemorySections,
  applyMemoryDiff,
  enforceLineBudget,
} from '../memory-file'

describe('memory-file', () => {
  test('ensureMemoryFile creates template if missing', async () => {
    const home = mkdtempSync(join(tmpdir(), 'mem-file-home-'))
    process.env.HOME = home
    const cwd = mkdtempSync(join(tmpdir(), 'mem-file-'))
    const storage = await createStorageClient<MagnitudeSlot>({ cwd })
    const p = await ensureMemoryFile(storage)
    const text = await readMemory(storage)
    expect(p.endsWith(MEMORY_RELATIVE_PATH)).toBe(true)
    expect(text).toBe(MEMORY_TEMPLATE)
    await rm(home, { recursive: true, force: true })
    await rm(cwd, { recursive: true, force: true })
  })

  test('parse/apply add update delete semantics', () => {
    const initial = `# Codebase
- use named exports

# Workflow
- ask clarifying questions when requirements are ambiguous

- ask before running long tests
`
    const { updated } = applyMemoryDiff(initial, {
      additions: [{ category: 'workflow', content: 'run reviewer after builder tasks' }],
      updates: [{ existing: 'use named exports', replacement: 'prefer named exports except React page defaults' }],
      deletions: [{ existing: 'ask clarifying questions when requirements are ambiguous' }],
    })

    const parsed = parseMemorySections(updated)
    expect(parsed.codebase).toContain('prefer named exports except React page defaults')
    expect(parsed.workflow).not.toContain('ask clarifying questions when requirements are ambiguous')
    expect(parsed.workflow).toContain('run reviewer after builder tasks')
  })

  test('idempotent diff application', () => {
    const first = applyMemoryDiff(MEMORY_TEMPLATE, {
      additions: [{ category: 'codebase', content: 'keep imports sorted' }],
    }).updated
    const second = applyMemoryDiff(first, {
      additions: [{ category: 'codebase', content: 'keep imports sorted' }],
    })
    expect(second.changed).toBe(false)
  })

  test('enforces line budget', () => {
    const lines = [
      '# Codebase',
      ...Array.from({ length: 120 }, (_, i) => `- codebase ${i + 1}`),
      '',
      '# Workflow',
      ...Array.from({ length: 120 }, (_, i) => `- workflow ${i + 1}`),
      '',
    ].join('\n')
    const out = enforceLineBudget(lines, 80)
    expect(out.split('\n').length).toBeLessThanOrEqual(80)
  })
})