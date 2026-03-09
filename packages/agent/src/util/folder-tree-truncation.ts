/**
 * Folder Tree Truncation
 *
 * Budget-aware truncation for folder trees to fit within token limits.
 * Shows folders only (no files), sorted by recency, with intelligent truncation.
 *
 * Based on: specs/25-01-19/folder-structure-awareness.md
 */

import { allocateBudget, type Measurement } from './truncation'
import type { FolderNode } from './tree'

// Token costs
const REMAINDER_COST = 5 // tokens for "... (N more)\n"
import { CHARS_PER_TOKEN } from '../constants'

export type { FolderNode }

/** Format a byte count as a human-readable token annotation */
function formatTokenAnnotation(totalBytes: number): string {
  if (totalBytes === 0) return ''
  const tokens = Math.round(totalBytes / CHARS_PER_TOKEN)
  if (tokens < 1000) return ` (~${tokens} tok)`
  return ` (~${Math.round(tokens / 1000)}k tok)`
}

/**
 * Cost of a single folder line (indentation + name + "/" + annotation + newline)
 */
function folderLineCost(depth: number, name: string, totalBytes: number = 0): number {
  const annotation = formatTokenAnnotation(totalBytes)
  return Math.ceil((depth * 2 + name.length + 1 + annotation.length + 1) / CHARS_PER_TOKEN)
}

/**
 * Generate indentation string
 */
function indent(depth: number): string {
  return '  '.repeat(depth)
}

/**
 * Generate a folder line with optional token annotation
 */
function folderLine(name: string, depth: number, totalBytes: number = 0): string {
  return indent(depth) + name + '/' + formatTokenAnnotation(totalBytes) + '\n'
}

/**
 * Sort folders by recency (most recently modified first)
 */
function sortByRecency(folders: readonly FolderNode[]): FolderNode[] {
  return [...folders].sort((a, b) => b.lastModified - a.lastModified)
}

/**
 * Measure the cost of a subtree (bounded - stops early if exceeds cap)
 */
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

/**
 * Render partial siblings when we can't fit all names
 */
function renderPartialSiblings(children: readonly FolderNode[], budget: number, depth: number): string {
  // children already sorted by recency
  let availableForNames = budget - REMAINDER_COST
  let result = ''
  let shown = 0

  for (const child of children) {
    const cost = folderLineCost(depth, child.name, child.totalBytes)
    if (cost > availableForNames) break
    result += folderLine(child.name, depth, child.totalBytes) // Name only, no subtree
    availableForNames -= cost
    shown++
  }

  if (shown < children.length) {
    const remaining = children.length - shown
    // Use "subfolders" when none shown, "more" when some were shown
    if (shown === 0) {
      result += indent(depth) + `... (${remaining} ${remaining === 1 ? 'subfolder' : 'subfolders'})\n`
    } else {
      result += indent(depth) + `... (${remaining} more)\n`
    }
  }

  return result
}

/**
 * Render children with budget allocation
 */
function renderChildrenWithBudget(children: readonly FolderNode[], budget: number, depth: number): string {
  if (children.length === 0 || budget <= 0) return ''

  // Sort by recency
  const sortedChildren = sortByRecency(children)

  // Calculate line costs for all siblings
  const lineCosts = sortedChildren.map(c => folderLineCost(depth, c.name, c.totalBytes))
  const totalLineCost = lineCosts.reduce((a, b) => a + b, 0)

  // PHASE 1: Can we fit all sibling names (without subtrees)?
  if (totalLineCost > budget) {
    // Can't fit all names - need to truncate with partial list
    return renderPartialSiblings(sortedChildren, budget, depth)
  }

  // PHASE 2: All names fit. Distribute remaining budget for subtrees.
  const subtreeBudget = budget - totalLineCost

  if (subtreeBudget <= 0) {
    // No budget for subtrees - just show names
    return sortedChildren.map(c => folderLine(c.name, depth, c.totalBytes)).join('')
  }

  // Measure subtree costs (bounded measurement - stops early if exceeds cap)
  const measurements = sortedChildren.map(c =>
    measureSubtreeCost(c.children, depth + 1, subtreeBudget)
  )

  // Smallest-first allocation (reuse from budgetTruncation.ts)
  // Small branches get their full cost, large branches share remainder
  const allocations = allocateBudget(measurements, subtreeBudget)

  let result = ''
  for (let i = 0; i < sortedChildren.length; i++) {
    const child = sortedChildren[i]
    result += folderLine(child.name, depth, child.totalBytes)

    if (child.children.length === 0) continue

    if (allocations[i] > 0) {
      // Has budget - recurse and let it handle partial rendering naturally
      result += renderChildrenWithBudget(
        child.children,
        allocations[i],
        depth + 1
      )
    } else {
      // Zero budget - show collapse indicator as last resort
      const count = child.children.length
      result += indent(depth + 1) + `... (${count} ${count === 1 ? 'subfolder' : 'subfolders'})\n`
    }
  }

  return result
}

/**
 * Truncate a folder tree to fit within a token budget.
 *
 * @param rootChildren - Array of root-level folder nodes
 * @param budgetTokens - Maximum tokens to use (default 400)
 * @returns Formatted folder tree string
 */
export function truncateFolderTree(rootChildren: readonly FolderNode[], budgetTokens: number = 400): string {
  if (rootChildren.length === 0) return ''
  return renderChildrenWithBudget(rootChildren, budgetTokens, 0).trimEnd()
}

/**
 * Build a folder tree from a flat list of folders.
 *
 * @param folders - Flat list of folder records with id, name, parentId, updatedAt
 * @returns Array of root-level FolderNode
 */
export function buildFolderTree(
  folders: Array<{ id: string; name: string; parentId: string | null; updatedAt: number }>
): FolderNode[] {
  const nodeMap = new Map<string, FolderNode>()
  const roots: FolderNode[] = []

  // First pass: create all nodes
  for (const folder of folders) {
    nodeMap.set(folder.id, {
      name: folder.name,
      children: [],
      lastModified: folder.updatedAt,
      totalBytes: 0,
    })
  }

  // Second pass: build tree structure
  for (const folder of folders) {
    const node = nodeMap.get(folder.id)!
    if (folder.parentId && nodeMap.has(folder.parentId)) {
      nodeMap.get(folder.parentId)!.children.push(node)
    } else {
      roots.push(node)
    }
  }

  return roots
}
