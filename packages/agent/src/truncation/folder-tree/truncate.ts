/**
 * Folder Tree Truncation
 *
 * Budget-aware truncation for folder trees to fit within token limits.
 * Shows folders only (no files), sorted by recency, with intelligent truncation.
 */

import { allocateBudget, charsToTokensUpper } from '../budget'
import type { Measurement } from '../budget'
import type { FolderNode } from './tree'
import { CHARS_PER_TOKEN_UPPER } from '../../constants'

// Token costs
const REMAINDER_COST = 5 // tokens for "... (N more)\n"

/** Format a byte count as a human-readable token annotation */
function formatTokenAnnotation(totalBytes: number): string {
  if (totalBytes === 0) return ''
  const tokens = Math.round(totalBytes / CHARS_PER_TOKEN_UPPER)
  if (tokens < 1000) return ` (~${tokens} tok)`
  return ` (~${Math.round(tokens / 1000)}k tok)`
}

/**
 * Cost of a single folder line (indentation + name + "/" + annotation + newline)
 */
function folderLineCost(depth: number, name: string, totalBytes: number = 0): number {
  const annotation = formatTokenAnnotation(totalBytes)
  return charsToTokensUpper(depth * 2 + name.length + 1 + annotation.length + 1)
}

function indent(depth: number): string {
  return '  '.repeat(depth)
}

function folderLine(name: string, depth: number, totalBytes: number = 0): string {
  return indent(depth) + name + '/' + formatTokenAnnotation(totalBytes) + '\n'
}

function sortByRecency(folders: readonly FolderNode[]): FolderNode[] {
  return [...folders].sort((a, b) => b.lastModified - a.lastModified)
}

function measureSubtreeCost(children: FolderNode[], depth: number, cap: number): Measurement {
  let total = 0
  for (const child of children) {
    total += folderLineCost(depth, child.name, child.totalBytes)
    if (total > cap) return { size: cap, exceeded: true }
    const sub = measureSubtreeCost(child.children, depth + 1, cap - total)
    total += sub.size
    if (sub.exceeded) return { size: cap, exceeded: true }
  }
  return { size: total, exceeded: false }
}

function renderPartialSiblings(children: readonly FolderNode[], budget: number, depth: number): string {
  let availableForNames = budget - REMAINDER_COST
  let result = ''
  let shown = 0

  for (const child of children) {
    const cost = folderLineCost(depth, child.name, child.totalBytes)
    if (cost > availableForNames) break
    result += folderLine(child.name, depth, child.totalBytes)
    availableForNames -= cost
    shown++
  }

  if (shown < children.length) {
    const remaining = children.length - shown
    if (shown === 0) {
      result += indent(depth) + `... (${remaining} ${remaining === 1 ? 'subfolder' : 'subfolders'})\n`
    } else {
      result += indent(depth) + `... (${remaining} more)\n`
    }
  }

  return result
}

function renderChildrenWithBudget(children: readonly FolderNode[], budget: number, depth: number): string {
  if (children.length === 0 || budget <= 0) return ''

  const sortedChildren = sortByRecency(children)

  const lineCosts = sortedChildren.map(c => folderLineCost(depth, c.name, c.totalBytes))
  const totalLineCost = lineCosts.reduce((a, b) => a + b, 0)

  if (totalLineCost > budget) {
    return renderPartialSiblings(sortedChildren, budget, depth)
  }

  const subtreeBudget = budget - totalLineCost

  if (subtreeBudget <= 0) {
    return sortedChildren.map(c => folderLine(c.name, depth, c.totalBytes)).join('')
  }

  const measurements = sortedChildren.map(c =>
    measureSubtreeCost(c.children, depth + 1, subtreeBudget)
  )

  const allocations = allocateBudget(measurements, subtreeBudget)

  let result = ''
  for (let i = 0; i < sortedChildren.length; i++) {
    const child = sortedChildren[i]
    result += folderLine(child.name, depth, child.totalBytes)

    if (child.children.length === 0) continue

    if (allocations[i] > 0) {
      result += renderChildrenWithBudget(
        child.children,
        allocations[i],
        depth + 1
      )
    } else {
      const count = child.children.length
      result += indent(depth + 1) + `... (${count} ${count === 1 ? 'subfolder' : 'subfolders'})\n`
    }
  }

  return result
}

/**
 * Truncate a folder tree to fit within a token budget.
 */
export function truncateFolderTree(rootChildren: readonly FolderNode[], budgetTokens: number = 400): string {
  if (rootChildren.length === 0) return ''
  return renderChildrenWithBudget(rootChildren, budgetTokens, 0).trimEnd()
}
