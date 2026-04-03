import { Schema } from '@effect/schema'
import type { TaskTypeDefinition } from '../types'
import { TaskDefinitionSchema } from './schema'

import featureMd from './composite/feature.md' with { type: 'text' }
import bugMd from './composite/bug.md' with { type: 'text' }
import refactorMd from './composite/refactor.md' with { type: 'text' }
import researchMd from './leaf/research.md' with { type: 'text' }
import planMd from './leaf/plan.md' with { type: 'text' }
import implementMd from './leaf/implement.md' with { type: 'text' }
import reviewMd from './leaf/review.md' with { type: 'text' }
import otherMd from './generic/other.md' with { type: 'text' }
import approveMd from './user/approve.md' with { type: 'text' }
import groupMd from './composite/group.md' with { type: 'text' }
import scanMd from './leaf/scan.md' with { type: 'text' }
import diagnoseMd from './leaf/diagnose.md' with { type: 'text' }
import webTestMd from './leaf/web-test.md' with { type: 'text' }
import ideateMd from './leaf/ideate.md' with { type: 'text' }

type Marker = 'lead' | 'worker' | 'criteria'
type ParsedSections = { lead: string; worker?: string; criteria: string }

const FRONTMATTER_REGEX = /^---\n([\s\S]*?)\n---\n([\s\S]*)$/
const MARKER_REGEX = /<!--\s*@([a-z-]+)\s*-->/g
const MARKER_ORDER: readonly Marker[] = ['lead', 'worker', 'criteria']

function splitFrontmatter(raw: string): { frontmatter: unknown; body: string } {
  const match = raw.match(FRONTMATTER_REGEX)
  if (!match) {
    throw new Error('Task definition missing frontmatter block.')
  }

  const parsed = Bun.YAML.parse(match[1])
  if (!parsed || typeof parsed !== 'object') {
    throw new Error('Task definition frontmatter must decode to an object.')
  }

  return {
    frontmatter: parsed,
    body: match[2],
  }
}

function parseSections(body: string): ParsedSections {
  const markers: Array<{ marker: string; index: number; length: number }> = []
  for (const match of body.matchAll(MARKER_REGEX)) {
    markers.push({
      marker: match[1],
      index: match.index ?? 0,
      length: match[0].length,
    })
  }

  if (markers.length === 0) {
    throw new Error('Task definition missing section markers. Expected <!-- @lead --> and <!-- @criteria -->.')
  }

  const seen = new Set<string>()
  let lastOrder = -1
  const sections: Partial<Record<Marker, string>> = {}

  for (let i = 0; i < markers.length; i++) {
    const current = markers[i]
    const marker = current.marker as Marker
    const orderIndex = MARKER_ORDER.indexOf(marker)

    if (orderIndex === -1) {
      throw new Error(
        `Unknown task definition section marker "@${current.marker}". Allowed markers: @lead, @worker, @criteria.`
      )
    }

    if (seen.has(marker)) {
      throw new Error(`Duplicate task definition section marker "@${marker}".`)
    }
    seen.add(marker)

    if (orderIndex < lastOrder) {
      throw new Error(
        `Task definition section marker "@${marker}" is out of order. Expected order: @lead -> @worker -> @criteria.`
      )
    }
    lastOrder = orderIndex

    const contentStart = current.index + current.length
    const contentEnd = i + 1 < markers.length ? markers[i + 1].index : body.length
    const content = body.slice(contentStart, contentEnd).trim()
    sections[marker] = content
  }

  return {
    lead: sections.lead ?? '',
    worker: sections.worker,
    criteria: sections.criteria ?? '',
  }
}

export function decodeTaskDefinition(raw: string): TaskTypeDefinition {
  const { frontmatter, body } = splitFrontmatter(raw)
  const sections = parseSections(body)

  return Schema.decodeUnknownSync(TaskDefinitionSchema)({
    frontmatter,
    sections,
  })
}

export const featureTaskType = decodeTaskDefinition(featureMd)
export const bugTaskType = decodeTaskDefinition(bugMd)
export const refactorTaskType = decodeTaskDefinition(refactorMd)
export const researchTaskType = decodeTaskDefinition(researchMd)
export const planTaskType = decodeTaskDefinition(planMd)
export const implementTaskType = decodeTaskDefinition(implementMd)
export const reviewTaskType = decodeTaskDefinition(reviewMd)
export const otherTaskType = decodeTaskDefinition(otherMd)
export const approveTaskType = decodeTaskDefinition(approveMd)
export const groupTaskType = decodeTaskDefinition(groupMd)
export const scanTaskType = decodeTaskDefinition(scanMd)
export const diagnoseTaskType = decodeTaskDefinition(diagnoseMd)
export const webTestTaskType = decodeTaskDefinition(webTestMd)
export const ideateTaskType = decodeTaskDefinition(ideateMd)
