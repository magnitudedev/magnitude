import { Context, Effect, Layer } from 'effect'
import * as os from 'os'
import * as path from 'path'
import { loadSkillset, SkillsetNotFoundError, SkillsetReadError } from './loader'
import { SkillParseError } from './parser'
import type { Skillset } from './types'

export interface SkillsetResolverService {
  readonly resolve: (
    name: string,
  ) => Effect.Effect<Skillset, SkillsetNotFoundError | SkillsetReadError | SkillParseError>
}

export class SkillsetResolver extends Context.Tag('SkillsetResolver')<
  SkillsetResolver,
  SkillsetResolverService
>() {}

const tryLoad = (dir: string) =>
  loadSkillset(dir).pipe(
    Effect.catchTag('SkillsetNotFoundError', () => Effect.succeed(null)),
    Effect.catchTag('SkillsetReadError', () => Effect.succeed(null)),
  )

export const SkillsetResolverLive = Layer.succeed(SkillsetResolver, {
  resolve: (name: string) =>
    Effect.gen(function* () {
      const cwd = process.cwd()
      const home = os.homedir()

      const candidates = [
        path.join(cwd, '.magnitude', 'skillsets', name),
        path.join(home, '.magnitude', 'skillsets', name),
      ]

      for (const dir of candidates) {
        const result = yield* tryLoad(dir)
        if (result !== null) return result
      }

      // Local dev: read from source skillsets directory
      const sourceDir = path.join(cwd, 'skillsets', name)
      return yield* loadSkillset(sourceDir)
    }),
})
