import { describe, expect, it } from 'bun:test'
import { Effect, Layer, ManagedRuntime } from 'effect'
import { mkdirSync, rmSync } from 'node:fs'
import { join } from 'node:path'
import { tmpdir } from 'node:os'

import { ConfigStorage, ConfigStorageLive } from './index'
import { GlobalStorage, makeGlobalStorage } from '../services'

function makeTempRoot(): string {
  const root = join(tmpdir(), `magnitude-storage-test-${Date.now()}-${Math.random().toString(16).slice(2)}`)
  mkdirSync(root, { recursive: true })
  return root
}

describe('setProviderOptions', () => {
  it('supports updater semantics', async () => {
    const root = makeTempRoot()
    const globalLayer = Layer.succeed(
      GlobalStorage,
      GlobalStorage.of(makeGlobalStorage({ root })),
    )
    const layer = Layer.provide(ConfigStorageLive, globalLayer)
    const runtime = ManagedRuntime.make(layer)

    try {
      await runtime.runPromise(
        Effect.flatMap(ConfigStorage, (s) =>
          s.setProviderOptions('lmstudio', { baseUrl: 'http://localhost:1234/v1' }),
        ),
      )

      await runtime.runPromise(
        Effect.flatMap(ConfigStorage, (s) =>
          s.setProviderOptions('lmstudio', (current) => ({
            ...(current ?? {}),
            rememberedModelIds: ['qwen3:8b'],
          })),
        ),
      )

      const updated = await runtime.runPromise(
        Effect.flatMap(ConfigStorage, (s) => s.getProviderOptions('lmstudio')),
      )

      expect(updated?.baseUrl).toBe('http://localhost:1234/v1')
      expect(updated?.rememberedModelIds).toEqual(['qwen3:8b'])
    } finally {
      await runtime.dispose()
      rmSync(root, { recursive: true, force: true })
    }
  })

  it('removes provider options when updater returns undefined', async () => {
    const root = makeTempRoot()
    const globalLayer = Layer.succeed(
      GlobalStorage,
      GlobalStorage.of(makeGlobalStorage({ root })),
    )
    const layer = Layer.provide(ConfigStorageLive, globalLayer)
    const runtime = ManagedRuntime.make(layer)

    try {
      await runtime.runPromise(
        Effect.flatMap(ConfigStorage, (s) =>
          s.setProviderOptions('ollama', { baseUrl: 'http://localhost:11434/v1' }),
        ),
      )

      await runtime.runPromise(
        Effect.flatMap(ConfigStorage, (s) =>
          s.setProviderOptions('ollama', () => undefined),
        ),
      )

      const removed = await runtime.runPromise(
        Effect.flatMap(ConfigStorage, (s) => s.getProviderOptions('ollama')),
      )

      expect(removed).toBeUndefined()

      const full = await runtime.runPromise(
        Effect.flatMap(ConfigStorage, (s) => s.load()),
      )
      expect(full.providers?.ollama).toBeUndefined()
    } finally {
      await runtime.dispose()
      rmSync(root, { recursive: true, force: true })
    }
  })
})
