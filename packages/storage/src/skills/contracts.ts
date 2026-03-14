import { Context, Effect } from 'effect'

export interface SkillStorageShape {
  readonly ensureDir: () => Effect.Effect<string>
  readonly list: () => Effect.Effect<string[]>
  readonly read: (skillName: string) => Effect.Effect<string>
  readonly write: (skillName: string, content: string) => Effect.Effect<void>
  readonly remove: (skillName: string) => Effect.Effect<void>
}

export class SkillStorage extends Context.Tag('SkillStorage')<
  SkillStorage,
  SkillStorageShape
>() {}