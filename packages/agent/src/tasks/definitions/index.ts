import type { TaskTypeDefinition } from '../types'

import featureMd from './feature.md' with { type: 'text' }
import bugMd from './bug.md' with { type: 'text' }
import refactorMd from './refactor.md' with { type: 'text' }
import researchMd from './research.md' with { type: 'text' }
import planMd from './plan.md' with { type: 'text' }
import implementMd from './implement.md' with { type: 'text' }
import reviewMd from './review.md' with { type: 'text' }
import otherMd from './other.md' with { type: 'text' }
import approveMd from './approve.md' with { type: 'text' }
import groupMd from './group.md' with { type: 'text' }

interface RawFrontmatter {
  id: string
  label: string
  description: string
  allowedAssignees: string[]
}

function parseTaskDefinition(raw: string): TaskTypeDefinition {
  const match = raw.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/)
  if (!match) throw new Error('Task definition missing frontmatter')

  const frontmatter = Bun.YAML.parse(match[1]) as RawFrontmatter
  const strategy = match[2].trim()

  return {
    id: frontmatter.id,
    label: frontmatter.label,
    description: frontmatter.description,
    allowedAssignees: frontmatter.allowedAssignees as TaskTypeDefinition['allowedAssignees'],
    strategy,
  }
}

export const featureTaskType = parseTaskDefinition(featureMd)
export const bugTaskType = parseTaskDefinition(bugMd)
export const refactorTaskType = parseTaskDefinition(refactorMd)
export const researchTaskType = parseTaskDefinition(researchMd)
export const planTaskType = parseTaskDefinition(planMd)
export const implementTaskType = parseTaskDefinition(implementMd)
export const reviewTaskType = parseTaskDefinition(reviewMd)
export const otherTaskType = parseTaskDefinition(otherMd)
export const approveTaskType = parseTaskDefinition(approveMd)
export const groupTaskType = parseTaskDefinition(groupMd)
