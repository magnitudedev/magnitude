import * as FileSystem from '@effect/platform/FileSystem'
import * as Path from '@effect/platform/Path'
import { type PlatformError } from '@effect/platform/Error'
import { Effect } from 'effect'

import { makeStorageIo } from '../io/storage'
import { ProjectStorage } from '../services'
import type { MemoryStorageShape } from './contracts'

export function makeMemoryStorage(): Effect.Effect<
  MemoryStorageShape,
  never,
  FileSystem.FileSystem | Path.Path | ProjectStorage
> {
  return Effect.gen(function* () {
    const io = yield* makeStorageIo()
    const projectStorage = yield* ProjectStorage
    const p = projectStorage.paths

    return {
      ensureFile: (template = '') =>
        Effect.gen(function* () {
          if (yield* io.pathExists(p.memoryFile)) {
            return p.memoryFile
          }
          yield* io.writeTextFile(p.memoryFile, template)
          return p.memoryFile
        }),

      read: () => io.readTextFile(p.memoryFile),

      write: (content) =>
        Effect.gen(function* () {
          yield* io.ensureParentDir(p.memoryFile)
          yield* io.writeTextFile(p.memoryFile, content)
        }),
    }
  })
}
