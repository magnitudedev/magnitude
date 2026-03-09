/**
 * Agent Skills Scanner
 *
 * Discovers and loads metadata from SKILL.md files following the Agent Skills spec.
 * Uses progressive disclosure: only parses frontmatter (name + description) at startup.
 * Full SKILL.md content is loaded on-demand by the agent when it activates a skill.
 *
 * Scan locations (in order, later overrides earlier on name conflicts):
 *   1. ~/.magnitude/skills/   (user-global, Magnitude-native)
 *   2. ~/.agents/skills/      (user-global, cross-agent standard)
 *   3. <cwd>/.magnitude/skills/ (project-local, Magnitude-native)
 *   4. <cwd>/.agents/skills/    (project-local, cross-agent standard)
 */

import { homedir } from 'os'
import { join } from 'path'
import { parseFrontmatter } from './frontmatter'

// =============================================================================
// Types
// =============================================================================

export interface SkillMetadata {
  readonly name: string
  readonly description: string
  readonly trigger: string
  readonly path: string  // absolute path to SKILL.md
}

// =============================================================================
// Directory Scanner
// =============================================================================

/**
 * Scan a single directory for SKILL.md files using Bun.Glob.
 * Returns discovered skills, or empty array if directory doesn't exist.
 */
async function scanDirectory(directory: string): Promise<SkillMetadata[]> {
  const skills: SkillMetadata[] = []
  const glob = new Bun.Glob('**/SKILL.md')

  try {
    for await (const match of glob.scan({
      cwd: directory,
      absolute: true,
      onlyFiles: true,
      followSymlinks: true,
    })) {
      try {
        const content = await Bun.file(match).text()
        const result = parseFrontmatter<{ name: string; description: string; trigger: string }>(content)
        if (!result) continue

        const { name, description, trigger } = result.frontmatter
        if (!name || !description || !trigger) continue

        skills.push({ name, description, trigger, path: match })
      } catch {
        // Skip files that can't be read
        continue
      }
    }
  } catch {
    // Directory doesn't exist or can't be scanned — skip silently
  }

  return skills
}

// =============================================================================
// Main Scanner
// =============================================================================

/**
 * Scan all skill directories and return deduplicated metadata.
 * Directories are scanned in order: global first, then project-local.
 * Later entries override earlier ones on name conflicts (project-local wins).
 */
export async function scanSkills(cwd: string): Promise<SkillMetadata[]> {
  const home = homedir()

  const directories = [
    join(home, '.agents', 'skills'),       // user-global (cross-agent)
    join(home, '.magnitude', 'skills'),    // user-global (Magnitude — overrides cross-agent)
    join(cwd, '.agents', 'skills'),        // project-local (cross-agent)
    join(cwd, '.magnitude', 'skills'),     // project-local (Magnitude — highest priority)
  ]

  // Scan all directories in parallel
  const results = await Promise.all(directories.map(scanDirectory))

  // Deduplicate by name — later entries override earlier ones
  const skillMap = new Map<string, SkillMetadata>()
  for (const dirSkills of results) {
    for (const skill of dirSkills) {
      skillMap.set(skill.name, skill)
    }
  }

  return Array.from(skillMap.values())
}
