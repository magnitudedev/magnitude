import { Data, Effect } from 'effect'
import { readdirSync } from 'fs'
import * as path from 'path'
import { parseSkill, SkillParseError } from './parser'
import type { Skill, Skillset } from './types'

export class SkillsetNotFoundError extends Data.TaggedError('SkillsetNotFoundError')<{
  readonly dir: string
}> {}

export class SkillsetReadError extends Data.TaggedError('SkillsetReadError')<{
  readonly path: string
  readonly cause: unknown
}> {}

export const loadSkillset = (
  dir: string,
): Effect.Effect<Skillset, SkillsetNotFoundError | SkillsetReadError | SkillParseError> =>
  Effect.gen(function* () {
    const skillsetMdPath = path.join(dir, 'SKILLSET.md')

    const exists = yield* Effect.tryPromise({
      try: () => Bun.file(skillsetMdPath).exists(),
      catch: (e) => new SkillsetReadError({ path: dir, cause: e }),
    })
    if (!exists) return yield* new SkillsetNotFoundError({ dir })

    const content = yield* Effect.tryPromise({
      try: () => Bun.file(skillsetMdPath).text(),
      catch: (e) => new SkillsetReadError({ path: skillsetMdPath, cause: e }),
    })

    const skillsDir = path.join(dir, 'skills')
    const entries = yield* Effect.try({
      try: () => readdirSync(skillsDir, { withFileTypes: true })
        .filter((d) => d.isDirectory())
        .map((d) => d.name),
      catch: (e) => new SkillsetReadError({ path: skillsDir, cause: e }),
    })

    const skillEntries = yield* Effect.all(
      entries.map((entry) =>
        Effect.gen(function* () {
          const skillMdPath = path.join(skillsDir, entry, 'SKILL.md')
          const skillContent = yield* Effect.tryPromise({
            try: () => Bun.file(skillMdPath).text(),
            catch: (e) => new SkillsetReadError({ path: skillMdPath, cause: e }),
          })
          const parsed = yield* parseSkill(skillContent)
          const skill = { ...parsed, path: path.join(skillsDir, entry) } satisfies Skill
          return [entry, skill] as const
        }),
      ),
      { concurrency: 'unbounded' },
    )

    return { path: dir, content, skills: Object.fromEntries(skillEntries) } satisfies Skillset
  })
