import * as path from 'node:path'
import * as os from 'node:os'
import { Effect } from 'effect'
import { parseSkill } from './parser'
import type { Skill } from './types'

/**
 * Load skills from standard directories.
 * Scans 2 directories in priority order (later overrides earlier on name conflicts):
 * 1. ~/.magnitude/skills/ (global, Magnitude-native)
 * 2. <cwd>/.magnitude/skills/ (project-local, highest priority)
 */
export async function loadSkills(cwd: string): Promise<Map<string, Skill>> {
  const home = os.homedir()

  const dirs = [
    path.join(home, '.magnitude', 'skills'),
    path.join(cwd, '.magnitude', 'skills'),
  ]

  const skills = new Map<string, Skill>()

  for (const dir of dirs) {
    // Check if directory exists
    try {
      const stat = await Bun.file(dir).stat()
      if (!stat.isDirectory()) continue
    } catch {
      continue
    }

    // Scan for SKILL.md files
    const glob = new Bun.Glob('**/SKILL.md')
    for await (const filePath of glob.scan({ cwd: dir })) {
      // Extract skill name from directory (e.g., "plan/SKILL.md" -> "plan")
      const skillName = path.dirname(filePath)

      // Later directories override earlier ones on name conflicts (project-local wins)

      const fullPath = path.join(dir, filePath)
      const content = await Bun.file(fullPath).text()

      const parsed = await Effect.runPromise(
        parseSkill(content).pipe(
          Effect.catchAll(() => Effect.succeed(null)),
        ),
      )

      if (parsed === null) continue

      const skill: Skill = {
        ...parsed,
        path: fullPath,
      }

      skills.set(skillName, skill)
    }
  }

  return skills
}
