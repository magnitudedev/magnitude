import type { PlatformError } from '@effect/platform/Error'
import { Context, Effect } from 'effect'

export interface MemoryStorageShape {
  readonly ensureFile: (template?: string) => Effect.Effect<string, PlatformError>
  readonly read: () => Effect.Effect<string, PlatformError>
  readonly write: (content: string) => Effect.Effect<void, PlatformError>
}

export class MemoryStorage extends Context.Tag('MemoryStorage')<
  MemoryStorage,
  MemoryStorageShape
>() {}
