import { describe, it, expect } from 'vitest'
import { Effect } from 'effect'
import { rm } from 'fs/promises'
import { tmpdir } from 'os'
import { join } from 'path'
import { persistResult, loadResult, hasResult, PersistError } from '../result-persistence'

const tmpDir = () => join(tmpdir(), `magnitude-persist-test-${Date.now()}-${Math.random().toString(36).slice(2)}`)

describe('result-persistence', () => {
  it('round-trips a simple object', async () => {
    const dir = tmpDir()
    try {
      const output = { mode: 'completed', exitCode: 0, stdout: 'hello\nworld' }
      await Effect.runPromise(persistResult(output, 'turn1', 'call1', dir))
      const loaded = await Effect.runPromise(loadResult('turn1', 'call1', dir))
      expect(loaded).toEqual(output)
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })

  it('hasResult returns false for missing, true after persist', async () => {
    const dir = tmpDir()
    try {
      const before = await Effect.runPromise(hasResult('turn2', 'call2', dir))
      expect(before).toBe(false)

      await Effect.runPromise(persistResult({ x: 1 }, 'turn2', 'call2', dir))

      const after = await Effect.runPromise(hasResult('turn2', 'call2', dir))
      expect(after).toBe(true)
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })

  it('loadResult fails with PersistError for missing file', async () => {
    const dir = tmpDir()
    const result = await Effect.runPromise(
      Effect.either(loadResult('turn3', 'call3', dir))
    )
    expect(result._tag).toBe('Left')
    if (result._tag === 'Left') {
      expect(result.left).toBeInstanceOf(PersistError)
      expect((result.left as PersistError).operation).toBe('load')
    }
  })

  it('round-trips nested/array values', async () => {
    const dir = tmpDir()
    try {
      const output = { items: [{ path: 'a.ts', depth: 0 }, { path: 'b.ts', depth: 1 }] }
      await Effect.runPromise(persistResult(output, 'turn4', 'call4', dir))
      const loaded = await Effect.runPromise(loadResult('turn4', 'call4', dir))
      expect(loaded).toEqual(output)
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })

  it('creates resultsDir if it does not exist', async () => {
    const baseDir = tmpDir()
    const dir = join(baseDir, 'nested', 'subdir')
    try {
      await Effect.runPromise(persistResult('hello', 'turn5', 'call5', dir))
      const loaded = await Effect.runPromise(loadResult('turn5', 'call5', dir))
      expect(loaded).toBe('hello')
    } finally {
      await rm(baseDir, { recursive: true, force: true })
    }
  })
})
