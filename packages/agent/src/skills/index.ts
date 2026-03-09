/**
 * Skill Registry
 *
 * Manages core (built-in) skills and resolution against user-provided skills.
 * Core skills are always available; user SKILL.md files with the same name override them.
 */

import type { SkillMetadata } from '../util/skill-scanner'
import { parseFrontmatter } from '../util/frontmatter'
import { FEATURE_SKILL } from './core/feature'
import { BUG_SKILL } from './core/bug'
import { REFACTOR_SKILL } from './core/refactor'

// =============================================================================
// Types
// =============================================================================

export interface CoreSkillEntry {
  readonly name: string
  readonly description: string
  readonly trigger: string
  readonly content: string
}

export interface ResolvedSkill {
  readonly name: string
  readonly description: string
  readonly content: string
  readonly source: 'core' | 'user'
}

// =============================================================================
// Core Skills Map
// =============================================================================

const CORE_SKILLS = new Map<string, CoreSkillEntry>([
  // Core skills disabled — prompts retained in ./core/ but not registered
])

export const CORE_SKILL_NAMES = [] as const
export type CoreSkillName = typeof CORE_SKILL_NAMES[number]

// =============================================================================
// Resolution
// =============================================================================

/**
 * Resolve a skill by name.
 * User skills override core skills when they share a name.
 * For user skills, reads the SKILL.md file from disk and strips frontmatter.
 */
export async function resolveSkill(
  name: string,
  userSkills: readonly SkillMetadata[]
): Promise<ResolvedSkill | null> {
  // User overrides take priority
  const userSkill = userSkills.find(s => s.name === name)
  if (userSkill) {
    const content = await readSkillContent(userSkill.path)
    if (content !== null) {
      return {
        name: userSkill.name,
        description: userSkill.description,
        content,
        source: 'user'
      }
    }
  }

  // Fall back to core skills
  const core = CORE_SKILLS.get(name)
  if (core) {
    return {
      name: core.name,
      description: core.description,
      content: core.content,
      source: 'core'
    }
  }

  return null
}

/**
 * Get core skills that are NOT overridden by user skills.
 * Used to build the <core_skills> section in the system prompt.
 */
export function getActiveCoreSkills(
  userSkills: readonly SkillMetadata[] | null
): CoreSkillEntry[] {
  const userNames = new Set((userSkills ?? []).map(s => s.name))
  const active: CoreSkillEntry[] = []

  for (const [name, skill] of CORE_SKILLS) {
    if (!userNames.has(name)) {
      active.push(skill)
    }
  }

  return active
}

/**
 * Get user skills that are NOT core skill overrides.
 * Used to build the <available_skills> section in the system prompt.
 */
export function getUserSkills(
  userSkills: readonly SkillMetadata[] | null
): readonly SkillMetadata[] {
  if (!userSkills) return []
  return userSkills.filter(s => !CORE_SKILLS.has(s.name))
}

/**
 * Get user skills that override core skills.
 */
export function getCoreOverrides(
  userSkills: readonly SkillMetadata[] | null
): readonly SkillMetadata[] {
  if (!userSkills) return []
  return userSkills.filter(s => CORE_SKILLS.has(s.name))
}

// =============================================================================
// Helpers
// =============================================================================

/**
 * Read a SKILL.md file and strip frontmatter, returning just the body content.
 */
async function readSkillContent(path: string): Promise<string | null> {
  try {
    const raw = await Bun.file(path).text()
    const result = parseFrontmatter(raw)
    if (result) {
      return result.body
    }
    // No frontmatter — return the whole file
    return raw.trim()
  } catch {
    return null
  }
}
