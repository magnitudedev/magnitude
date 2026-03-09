import type { CheckResult } from '../../types'
import type { MemoryCategory, MemoryDiffResult } from './types'

const VALID_CATEGORIES = new Set<MemoryCategory>(['codebase', 'workflow'])
const NARRATIVE_PREFIXES = ['the user', 'this session', 'the assistant', 'it seems', 'they prefer']

export function parseDiff(raw: string): { diff: MemoryDiffResult | null; error?: string } {
  try {
    const obj = JSON.parse(raw)
    if (!obj || typeof obj !== 'object') return { diff: null, error: 'Not a JSON object' }
    const diff: MemoryDiffResult = {
      reasoning: typeof obj.reasoning === 'string' ? obj.reasoning : '',
      additions: Array.isArray(obj.additions) ? obj.additions : [],
      updates: Array.isArray(obj.updates) ? obj.updates : [],
      deletions: Array.isArray(obj.deletions) ? obj.deletions : [],
    }
    return { diff }
  } catch (error) {
    return { diff: null, error: `Invalid JSON: ${error instanceof Error ? error.message : String(error)}` }
  }
}

export function validJsonObject(diff: MemoryDiffResult | null, error?: string): CheckResult {
  if (!diff) return { passed: false, message: error ?? 'Invalid diff JSON' }
  return { passed: true }
}

export function validCategories(diff: MemoryDiffResult): CheckResult {
  const bad = diff.additions.find((a) => !VALID_CATEGORIES.has(a.category))
  if (bad) return { passed: false, message: `Invalid category: ${String(bad.category)}` }
  return { passed: true }
}

export function imperativeLineShape(diff: MemoryDiffResult): CheckResult {
  const lines = [...diff.additions.map((a) => a.content), ...diff.updates.map((u) => u.replacement)]
    .map((s) => (s || '').trim().toLowerCase())
    .filter(Boolean)

  const narrative = lines.find((line) => NARRATIVE_PREFIXES.some((prefix) => line.startsWith(prefix)))
  if (narrative) return { passed: false, message: `Narrative/non-imperative line: "${narrative}"` }
  return { passed: true }
}

export function exactEmpty(diff: MemoryDiffResult): CheckResult {
  const passed = diff.additions.length === 0 && diff.updates.length === 0 && diff.deletions.length === 0
  return { passed, message: passed ? undefined : 'Expected exact empty diff' }
}

export function operationBounds(diff: MemoryDiffResult, min?: number, max?: number): CheckResult {
  const total = diff.additions.length + diff.updates.length + diff.deletions.length
  if (typeof min === 'number' && total < min) return { passed: false, message: `Too few operations: ${total} < ${min}` }
  if (typeof max === 'number' && total > max) return { passed: false, message: `Too many operations: ${total} > ${max}` }
  return { passed: true }
}

export function requiredCategories(diff: MemoryDiffResult, categories: MemoryCategory[]): CheckResult {
  const missing = categories.filter((category) => !diff.additions.some((a) => a.category === category))
  if (missing.length) return { passed: false, message: `Missing required addition categories: ${missing.join(', ')}` }
  return { passed: true }
}

export function allowedCategories(diff: MemoryDiffResult, categories: MemoryCategory[]): CheckResult {
  const disallowed = diff.additions.filter((a) => !categories.includes(a.category))
  if (disallowed.length) return { passed: false, message: `Disallowed addition categories: ${[...new Set(disallowed.map((d) => d.category))].join(', ')}` }
  return { passed: true }
}

function normalizeLine(s: string): string {
  return s.toLowerCase().replace(/[^\w\s]/g, ' ').replace(/\s+/g, ' ').trim()
}

function parseMemoryLines(currentMemory: string): string[] {
  return currentMemory
    .split('\n')
    .map((l) => l.trim())
    .filter((l) => l.startsWith('- '))
    .map((l) => l.slice(2).trim())
    .filter(Boolean)
}

function overlap(a: string, b: string): number {
  const as = new Set(a.split(' ').filter(Boolean))
  const bs = new Set(b.split(' ').filter(Boolean))
  if (!as.size || !bs.size) return 0
  let inter = 0
  for (const w of as) if (bs.has(w)) inter++
  return inter / Math.min(as.size, bs.size)
}

export function duplicateDetection(diff: MemoryDiffResult, currentMemory: string): CheckResult {
  const existing = parseMemoryLines(currentMemory).map(normalizeLine).filter(Boolean)
  for (const add of diff.additions) {
    const n = normalizeLine(add.content || '')
    if (!n) continue
    const dup = existing.find((e) => e === n || e.includes(n) || n.includes(e) || overlap(e, n) >= 0.8)
    if (dup) return { passed: false, message: `Near-duplicate addition detected: "${add.content}"` }
  }
  return { passed: true }
}

export function hasUpdateOrDeletion(diff: MemoryDiffResult): CheckResult {
  const total = diff.updates.length + diff.deletions.length
  if (total === 0) return { passed: false, message: 'Expected at least one update or deletion operation' }
  return { passed: true }
}