import { describe, test, expect } from 'bun:test'
import { mkdtempSync } from 'fs'
import { rm } from 'fs/promises'
import { join } from 'path'
import { tmpdir } from 'os'
import { Effect, Layer } from 'effect'
import { BunFileSystem, BunPath } from '@effect/platform-bun'
import {
  StorageLive,
  GlobalStorageLive,
  VersionLive,
  ProjectStorageLiveFromCwd,
  MagnitudeStorage,
} from '@magnitudedev/storage'

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
    const prevHome = process.env.HOME
    process.env.HOME = home
    const cwd = mkdtempSync(join(tmpdir(), 'mem-file-'))

    const testLayer = Layer.provide(
      StorageLive,
      Layer.mergeAll(
        GlobalStorageLive,
        VersionLive('test'),
        ProjectStorageLiveFromCwd(cwd),
        BunFileSystem.layer,
        BunPath.layer,
      ),
    )

    const [p, text] = await Effect.runPromise(
      Effect.gen(function* () {
        const storage = yield* MagnitudeStorage
        const p = yield* ensureMemoryFile(storage)
        const text = yield* readMemory(storage)
        return [p, text] as const
      }).pipe(Effect.provide(testLayer))
    )

    expect(p.endsWith(MEMORY_RELATIVE_PATH)).toBe(true)
    expect(text).toBe(MEMORY_TEMPLATE)

    process.env.HOME = prevHome
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
