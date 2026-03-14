import { Effect, Layer } from 'effect'

import { ProjectStorage } from '../services'
import { MemoryStorage } from './contracts'
import {
  ensureMemoryFile,
  readMemory,
  writeMemory,
} from './storage'

export const MemoryStorageLive = Layer.effect(
  MemoryStorage,
  Effect.gen(function* () {
    const projectStorage = yield* ProjectStorage

    return MemoryStorage.of({
      ensureFile: (template?: string) =>
        Effect.promise(() =>
          ensureMemoryFile(projectStorage.paths, template)
        ),
      read: () => Effect.promise(() => readMemory(projectStorage.paths)),
      write: (content: string) =>
        Effect.promise(() => writeMemory(projectStorage.paths, content)),
    })
  })
)