import { Context, Effect, Layer } from 'effect'
import * as fs from 'fs'
import * as os from 'os'
import * as path from 'path'
import { loadSkillset, SkillsetNotFoundError, SkillsetReadError } from './loader'
import { SkillParseError } from './parser'
import type { Skillset, SkillsetInfo } from './types'

export interface SkillsetResolverService {
  readonly resolve: (
    name: string,
  ) => Effect.Effect<Skillset, SkillsetNotFoundError | SkillsetReadError | SkillParseError>
  readonly list: () => Effect.Effect<SkillsetInfo[]>
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

const listSkillsetsInDir = (dir: string, scope: 'project' | 'global'): Effect.Effect<SkillsetInfo[]> =>
  Effect.try({
    try: () => {
      if (!fs.existsSync(dir)) return []
      const entries = fs.readdirSync(dir, { withFileTypes: true })
      const infos: SkillsetInfo[] = []
      for (const entry of entries) {
        if (!entry.isDirectory()) continue
        const skillsetPath = path.join(dir, entry.name)
        const markerPath = path.join(skillsetPath, 'SKILLSET.md')
        if (fs.existsSync(markerPath)) {
          infos.push({ name: entry.name, path: skillsetPath, scope })
        }
      }
      return infos
    },
    catch: () => [] as SkillsetInfo[],
  }).pipe(Effect.orElseSucceed(() => []))

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

      return yield* new SkillsetNotFoundError({ dir: candidates.join(', ') })
    }),

  list: () =>
    Effect.gen(function* () {
      const cwd = process.cwd()
      const home = os.homedir()

      const projectDir = path.join(cwd, '.magnitude', 'skillsets')
      const globalDir = path.join(home, '.magnitude', 'skillsets')

      const [projectInfos, globalInfos] = yield* Effect.all([
        listSkillsetsInDir(projectDir, 'project'),
        listSkillsetsInDir(globalDir, 'global'),
      ])

      // Deduplicate: project takes precedence over global
      const seen = new Set<string>()
      const result: SkillsetInfo[] = []
      for (const info of [...projectInfos, ...globalInfos]) {
        if (!seen.has(info.name)) {
          seen.add(info.name)
          result.push(info)
        }
      }
      return result
    }),
})
