/**
 * Skill Registry
 *
 * Resolves user-provided skills discovered from SKILL.md files.
 */

import { copyFile, mkdir, readdir } from 'node:fs/promises'
import { basename, dirname, join } from 'node:path'
import type { SkillMetadata } from '../util/skill-scanner'


// =============================================================================
// Types
// =============================================================================

export interface ResolvedSkill extends SkillMetadata {
  readonly content: string
}

// =============================================================================
// Resolution
// =============================================================================

/**
 * Resolve a skill by name.
 * For user skills, reads the SKILL.md file from workspace copy and strips frontmatter.
 */
export async function resolveSkill(
  name: string,
  userSkills: readonly SkillMetadata[]
): Promise<ResolvedSkill | null> {
  const userSkill = userSkills.find(s => s.name === name)
  if (!userSkill) return null

  const workspaceSkillPath = await copySkillToWorkspace(userSkill)
  const content = await readSkillContent(workspaceSkillPath)
  if (content === null) return null

  return {
    name: userSkill.name,
    description: userSkill.description,
    content,
    path: workspaceSkillPath
  }
}

export function getUserSkills(
  userSkills: readonly SkillMetadata[] | null
): readonly SkillMetadata[] {
  return userSkills ?? []
}

// =============================================================================
// Helpers
// =============================================================================

/**
 * Copy a user skill directory into workspace and return the copied SKILL.md path.
 */
async function copySkillToWorkspace(skill: SkillMetadata): Promise<string> {
  const workspaceRoot = process.env.M
  if (!workspaceRoot) return skill.path

  const sourceDir = dirname(skill.path)
  const destinationDir = join(workspaceRoot, 'skills', skill.name)

  await copyDirectory(sourceDir, destinationDir)

  return join(destinationDir, basename(skill.path))
}

async function copyDirectory(sourceDir: string, destinationDir: string): Promise<void> {
  await mkdir(destinationDir, { recursive: true })
  const entries = await readdir(sourceDir, { withFileTypes: true })

  for (const entry of entries) {
    const sourcePath = join(sourceDir, entry.name)
    const destinationPath = join(destinationDir, entry.name)

    if (entry.isDirectory()) {
      await copyDirectory(sourcePath, destinationPath)
    } else if (entry.isFile()) {
      await mkdir(dirname(destinationPath), { recursive: true })
      await copyFile(sourcePath, destinationPath)
    }
  }
}

/**
 * Read a SKILL.md file and return the full raw content (including frontmatter).
 */
async function readSkillContent(path: string): Promise<string | null> {
  try {
    return await Bun.file(path).text()
  } catch {
    return null
  }
}
