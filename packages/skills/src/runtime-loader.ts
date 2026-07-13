import * as path from 'node:path'
import * as os from 'node:os'
import { Effect } from 'effect'
import { parseSkill } from './parser'
import type { ParsedSkill, Skill } from './types'

export type SkillLoadDiagnostic =
  | { readonly type: 'directory_stat_failed'; readonly dir: string; readonly error: unknown }
  | { readonly type: 'directory_scan_failed'; readonly dir: string; readonly error: unknown }
  | { readonly type: 'skill_file_read_failed'; readonly dir: string; readonly filePath: string; readonly fullPath: string; readonly error: unknown }
  | { readonly type: 'skill_parse_failed'; readonly dir: string; readonly filePath: string; readonly fullPath: string; readonly error: unknown }

export interface LoadSkillsOptions {
  readonly onDiagnostic?: (diagnostic: SkillLoadDiagnostic) => void
}

const errorMessage = (error: unknown): string =>
  error instanceof Error ? error.message : String(error)

const errorStack = (error: unknown): string | undefined =>
  error instanceof Error ? error.stack : undefined

const isNotFoundError = (error: unknown): boolean =>
  typeof error === 'object'
  && error !== null
  && 'code' in error
  && (error as { code?: unknown }).code === 'ENOENT'

export function skillLoadDiagnosticLogFields(
  diagnostic: SkillLoadDiagnostic,
): Record<string, unknown> {
  return {
    type: diagnostic.type,
    dir: diagnostic.dir,
    ...('filePath' in diagnostic ? { filePath: diagnostic.filePath, fullPath: diagnostic.fullPath } : {}),
    error: errorMessage(diagnostic.error),
    stack: errorStack(diagnostic.error),
  }
}

function reportSkillLoadDiagnostic(
  options: LoadSkillsOptions | undefined,
  diagnostic: SkillLoadDiagnostic,
): void {
  if (options?.onDiagnostic) {
    options.onDiagnostic(diagnostic)
    return
  }
  const pathInfo = 'fullPath' in diagnostic ? ` file=${diagnostic.fullPath}` : ` dir=${diagnostic.dir}`
  console.warn(`[skills] ${diagnostic.type}${pathInfo}: ${errorMessage(diagnostic.error)}`)
}

/**
 * Load skills from standard directories.
 * Scans 6 directories in priority order (later overrides earlier on name conflicts):
 * 1. ~/.claude/skills/ (global, Claude Code)
 * 2. ~/.agents/skills/ (global, cross-agent standard)
 * 3. ~/.magnitude/skills/ (global, Magnitude-native)
 * 4. <cwd>/.claude/skills/ (project-local, Claude Code)
 * 5. <cwd>/.agents/skills/ (project-local, cross-agent standard)
 * 6. <cwd>/.magnitude/skills/ (project-local, highest priority)
 */
export async function loadSkills(cwd: string, options?: LoadSkillsOptions): Promise<Map<string, Skill>> {
  const home = os.homedir()

  const dirs = [
    path.join(home, '.claude', 'skills'),
    path.join(home, '.agents', 'skills'),
    path.join(home, '.magnitude', 'skills'),
    path.join(cwd, '.claude', 'skills'),
    path.join(cwd, '.agents', 'skills'),
    path.join(cwd, '.magnitude', 'skills'),
  ]

  const skills = new Map<string, Skill>()

  for (const dir of dirs) {
    // Check if directory exists
    try {
      const stat = await Bun.file(dir).stat()
      if (!stat.isDirectory()) continue
    } catch (error) {
      if (!isNotFoundError(error)) {
        reportSkillLoadDiagnostic(options, { type: 'directory_stat_failed', dir, error })
      }
      continue
    }

    // Scan for SKILL.md files
    const glob = new Bun.Glob('**/SKILL.md')
    try {
      for await (const filePath of glob.scan({ cwd: dir })) {
        // Extract skill name from directory (e.g., "plan/SKILL.md" -> "plan")
        const skillName = path.dirname(filePath)

        // Later directories override earlier ones on name conflicts (project-local wins)

        const fullPath = path.join(dir, filePath)
        let content: string
        try {
          content = await Bun.file(fullPath).text()
        } catch (error) {
          reportSkillLoadDiagnostic(options, { type: 'skill_file_read_failed', dir, filePath, fullPath, error })
          continue
        }

        let parsed: ParsedSkill
        try {
          parsed = await Effect.runPromise(parseSkill(content))
        } catch (error) {
          reportSkillLoadDiagnostic(options, { type: 'skill_parse_failed', dir, filePath, fullPath, error })
          continue
        }

        const skill: Skill = {
          ...parsed,
          path: fullPath,
        }

        skills.set(skillName, skill)
      }
    } catch (error) {
      reportSkillLoadDiagnostic(options, { type: 'directory_scan_failed', dir, error })
      continue
    }
  }

  return skills
}
