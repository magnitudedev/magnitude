import type { TaskTypeDefinition, TaskAssignee } from '../types'

/**
 * Parse a task type definition from a markdown file with YAML-like frontmatter.
 *
 * Expected format:
 * ---
 * id: feature
 * label: Feature
 * description: Deliver a user-facing capability
 * allowedAssignees: [builder]
 * ---
 *
 * # Feature
 * ...strategy body...
 */
export function parseTaskDefinition(raw: string): TaskTypeDefinition {
  const fmMatch = raw.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/)
  if (!fmMatch) {
    throw new Error('Task definition missing frontmatter (expected --- delimiters)')
  }

  const frontmatter = fmMatch[1]
  const strategy = fmMatch[2].trim()

  const id = extractField(frontmatter, 'id')
  const label = extractField(frontmatter, 'label')
  const description = extractField(frontmatter, 'description')
  const allowedAssignees = extractList(frontmatter, 'allowedAssignees') as TaskAssignee[]

  if (!id || !label || !description || allowedAssignees.length === 0) {
    throw new Error(`Task definition missing required frontmatter fields (id, label, description, allowedAssignees)`)
  }

  return { id, label, description, allowedAssignees, strategy }
}

function extractField(frontmatter: string, key: string): string {
  const match = frontmatter.match(new RegExp(`^${key}:\\s*(.+)$`, 'm'))
  return match ? match[1].trim() : ''
}

function extractList(frontmatter: string, key: string): string[] {
  const match = frontmatter.match(new RegExp(`^${key}:\\s*\\[([^\\]]*)]`, 'm'))
  if (!match) return []
  return match[1].split(',').map(s => s.trim()).filter(Boolean)
}
