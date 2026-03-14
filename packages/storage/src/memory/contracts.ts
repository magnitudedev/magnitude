import { Context, Effect } from 'effect'

export interface MemoryStorageShape {
  readonly ensureFile: (template?: string) => Effect.Effect<string>
  readonly read: () => Effect.Effect<string>
  readonly write: (content: string) => Effect.Effect<void>
}

export class MemoryStorage extends Context.Tag('MemoryStorage')<
  MemoryStorage,
  MemoryStorageShape
>() {}