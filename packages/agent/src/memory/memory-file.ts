import type { StorageClient } from '@magnitudedev/storage'

export const MEMORY_RELATIVE_PATH = '.magnitude/memory.md'

export const MEMORY_TEMPLATE = `# Codebase
- 


# Workflow
- 
`

export interface ParsedMemorySections {
  codebase: string[]

  workflow: string[]
}

export interface MemoryDiff {
  additions?: Array<{ category: keyof ParsedMemorySections; content: string }>
  updates?: Array<{ existing: string; replacement: string }>
  deletions?: Array<{ existing: string }>
}

const SECTION_HEADERS: Array<{ key: keyof ParsedMemorySections; title: string }> = [
  { key: 'codebase', title: 'Codebase' },

  { key: 'workflow', title: 'Workflow' },
]

function normalizeLine(line: string): string {
  return line.trim()
}

function normalizeBulletContent(line: string): string | null {
  const t = line.trim()
  if (!t.startsWith('-')) return null
  const content = t.slice(1).trim()
  if (!content) return null
  return content
}

function sectionKeyFromHeader(line: string): keyof ParsedMemorySections | null {
  const t = line.trim()
  if (!t.startsWith('# ')) return null
  const title = t.slice(2).trim().toLowerCase()
  if (title === 'codebase') return 'codebase'

  if (title === 'workflow') return 'workflow'
  return null
}

export async function ensureMemoryFile(storage: StorageClient): Promise<string> {
  await storage.memory.ensureFile(MEMORY_TEMPLATE)
  return storage.memory.getPath()
}

export async function readMemory(storage: StorageClient): Promise<string> {
  return await storage.memory.read()
}

export async function writeMemory(storage: StorageClient, content: string): Promise<void> {
  await storage.memory.write(content)
}

export function parseMemorySections(content: string): ParsedMemorySections {
  const out: ParsedMemorySections = {
    codebase: [],

    workflow: [],
  }

  const lines = content.split('\n')
  let current: keyof ParsedMemorySections | null = null

  for (const raw of lines) {
    const key = sectionKeyFromHeader(raw)
    if (key) {
      current = key
      continue
    }
    if (!current) continue
    const bullet = normalizeBulletContent(raw)
    if (bullet) out[current].push(bullet)
  }

  for (const key of Object.keys(out) as Array<keyof ParsedMemorySections>) {
    out[key] = Array.from(new Set(out[key].map(normalizeLine).filter(Boolean)))
  }

  return out
}

export function renderMemory(parsed: ParsedMemorySections): string {
  const blocks: string[] = []
  for (const section of SECTION_HEADERS) {
    blocks.push(`# ${section.title}`)
    const lines = parsed[section.key]
    if (lines.length === 0) {
      blocks.push('- ')
    } else {
      for (const line of lines) blocks.push(`- ${line}`)
    }
    blocks.push('')
  }
  return blocks.join('\n')
}

function findAndRemoveExact(parsed: ParsedMemorySections, existing: string): boolean {
  const target = normalizeLine(existing).replace(/^-+\s*/, '')
  let removed = false
  for (const key of Object.keys(parsed) as Array<keyof ParsedMemorySections>) {
    const next = parsed[key].filter((line) => {
      const keep = normalizeLine(line) !== target
      if (!keep) removed = true
      return keep
    })
    parsed[key] = next
  }
  return removed
}

function findAndReplaceExact(parsed: ParsedMemorySections, existing: string, replacement: string): boolean {
  const target = normalizeLine(existing).replace(/^-+\s*/, '')
  const nextValue = normalizeLine(replacement).replace(/^-+\s*/, '')
  let replaced = false
  for (const key of Object.keys(parsed) as Array<keyof ParsedMemorySections>) {
    parsed[key] = parsed[key].map((line) => {
      if (normalizeLine(line) === target) {
        replaced = true
        return nextValue
      }
      return line
    })
  }
  return replaced
}

export function applyMemoryDiff(content: string, diff: MemoryDiff): { updated: string; changed: boolean; warnings: string[] } {
  const parsed = parseMemorySections(content)
  const warnings: string[] = []

  for (const del of diff.deletions ?? []) {
    const ok = findAndRemoveExact(parsed, del.existing)
    if (!ok) warnings.push(`Deletion target not found: ${del.existing}`)
  }

  for (const upd of diff.updates ?? []) {
    const ok = findAndReplaceExact(parsed, upd.existing, upd.replacement)
    if (!ok) warnings.push(`Update target not found: ${upd.existing}`)
  }

  for (const add of diff.additions ?? []) {
    const text = normalizeLine(add.content).replace(/^-+\s*/, '')
    if (!text) continue
    const key = add.category
    if (!parsed[key].includes(text)) parsed[key].push(text)
  }

  for (const key of Object.keys(parsed) as Array<keyof ParsedMemorySections>) {
    parsed[key] = Array.from(new Set(parsed[key].map(normalizeLine).filter(Boolean)))
  }

  const updated = renderMemory(parsed)
  const changed = normalizeLine(updated) !== normalizeLine(content)
  return { updated, changed, warnings }
}

export function enforceLineBudget(content: string, maxLines = 150): string {
  const parsed = parseMemorySections(content)

  const totalSectionLines = () => renderMemory(parsed).split('\n').length
  if (totalSectionLines() <= maxLines) return renderMemory(parsed)

  const order: Array<keyof ParsedMemorySections> = ['codebase', 'workflow']
  while (totalSectionLines() > maxLines) {
    let removed = false
    for (const key of order) {
      if (parsed[key].length > 0) {
        parsed[key].shift()
        removed = true
        if (totalSectionLines() <= maxLines) break
      }
    }
    if (!removed) break
  }

  return renderMemory(parsed)
}