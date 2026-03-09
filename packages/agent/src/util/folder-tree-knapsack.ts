import { access } from 'fs/promises'
import { join } from 'path'
import { CHARS_PER_TOKEN } from '../constants'
import { createDefaultIgnore } from './gitignore'
import { runGitCommand } from './git-command'
import { walk, type Entry } from './walk'

const LAMBDA = Math.LN2 / 7
const DIR_STRUCTURAL_ALPHA = 0.3

interface KnapsackNode {
  id: number
  name: string
  relativePath: string
  kind: 'dir' | 'file'
  depth: number
  costTokens: number
  score: number
  children: KnapsackNode[]
}

type BuildNode = KnapsackNode & { parent?: BuildNode }

type NodeDP = {
  dp: Float64Array
  maxCost: number
}

const EPSILON = 1e-9

export function merge(A: Float64Array, B: Float64Array, W: number): Float64Array {
  const lenA = Math.min(A.length, W + 1)
  const lenB = Math.min(B.length, W + 1)
  const lenR = Math.min(lenA + lenB - 1, W + 1)
  const R = new Float64Array(lenR)
  for (let a = 0; a < lenA; a++) {
    if (A[a] === 0 && a > 0) continue
    for (let b = 0; b < lenB && a + b < lenR; b++) {
      const v = A[a] + B[b]
      if (v > R[a + b]) R[a + b] = v
    }
  }
  return R
}

function nowDays(): number {
  return Date.now() / 1000 / 86400
}

function normalizePath(p: string): string {
  return p.replaceAll('\\', '/')
}

function computeRawRecencyScore(timestampDays: number, currentDays: number): number {
  return Math.exp(-LAMBDA * (currentDays - timestampDays))
}

async function collectGitRawScores(cwd: string, currentDays: number): Promise<Map<string, number> | null> {
  try {
    await access(join(cwd, '.git'))
  } catch {
    return null
  }

  const output = await runGitCommand(
    ['log', '--max-count=200', '--name-only', '--format=%ct', '--diff-filter=ACDMR'],
    cwd,
    1000
  )
  if (output === null) return null

  const scores = new Map<string, number>()
  const lines = output.split('\n')
  let tsDays: number | null = null

  for (const rawLine of lines) {
    const line = rawLine.trim()
    if (!line) continue
    if (/^\d+$/.test(line)) {
      tsDays = Number(line) / 86400
      continue
    }
    if (tsDays === null) continue
    const file = normalizePath(line)
    const score = computeRawRecencyScore(tsDays, currentDays)
    scores.set(file, (scores.get(file) ?? 0) + score)
  }

  return scores
}

function lineCost(depth: number, name: string, isDir: boolean): number {
  // +1 for newline character
  const chars = depth * 2 + name.length + (isDir ? 2 : 1) + 1
  return Math.max(1, Math.ceil(chars / CHARS_PER_TOKEN))
}

function remainderLineCost(depth: number, childCount: number): number {
  // "  ".repeat(depth+1) + "... (N more)\n"
  const chars = (depth + 1) * 2 + 9 + String(childCount).length + 1
  return Math.ceil(chars / CHARS_PER_TOKEN)
}

function buildNodeTree(entries: Entry[]): BuildNode[] {
  const all = [...entries].sort((a, b) => a.relativePath.length - b.relativePath.length)
  const nodes = new Map<string, BuildNode>()
  const roots: BuildNode[] = []
  let id = 1

  for (const entry of all) {
    const node: BuildNode = {
      id: id++,
      name: entry.name,
      relativePath: normalizePath(entry.relativePath),
      kind: entry.type,
      depth: entry.depth,
      costTokens: lineCost(entry.depth, entry.name, entry.type === 'dir'),
      score: 0,
      children: []
    }
    nodes.set(node.relativePath, node)
  }

  for (const entry of all) {
    const path = normalizePath(entry.relativePath)
    const node = nodes.get(path)
    if (!node) continue

    const idx = path.lastIndexOf('/')
    if (idx < 0) {
      roots.push(node)
      continue
    }

    const parentPath = path.slice(0, idx)
    const parent = nodes.get(parentPath)
    if (!parent || parent.kind !== 'dir') {
      roots.push(node)
      continue
    }
    parent.children.push(node)
    node.parent = parent
  }

  roots.sort((a, b) => a.name.localeCompare(b.name))
  for (const node of nodes.values()) {
    node.children.sort((a, b) => a.name.localeCompare(b.name))
    if (node.kind === 'dir' && node.children.length > 0) {
      node.costTokens += remainderLineCost(node.depth, node.children.length)
    }
  }
  return roots
}

function applyScores(
  roots: BuildNode[],
  entries: Entry[],
  gitRaw: Map<string, number> | null,
  currentDays: number
): void {
  const fileRaw = new Map<string, number>()
  const maxFinder: number[] = []

  for (const entry of entries) {
    if (entry.type !== 'file') continue
    const path = normalizePath(entry.relativePath)
    let raw = 0
    if (gitRaw && gitRaw.has(path)) {
      raw = gitRaw.get(path) ?? 0
    } else if (entry.mtimeMs) {
      raw = computeRawRecencyScore(entry.mtimeMs / 1000 / 86400, currentDays)
    }
    fileRaw.set(path, raw)
    maxFinder.push(raw)
  }

  const maxRaw = Math.max(0, ...maxFinder)
  const nodeByPath = new Map<string, BuildNode>()
  const stack = [...roots]
  while (stack.length > 0) {
    const n = stack.pop()!
    nodeByPath.set(n.relativePath, n)
    for (const c of n.children) stack.push(c)
  }

  for (const [path, raw] of fileRaw) {
    const n = nodeByPath.get(path)
    if (!n || n.kind !== 'file') continue
    n.score = maxRaw > 0 ? Math.log1p(raw) / Math.log1p(maxRaw) : 0
  }

  function dfs(node: BuildNode): number {
    if (node.kind === 'file') return node.score
    let descFileScore = 0
    for (const child of node.children) {
      descFileScore += dfs(child)
    }
    node.score = DIR_STRUCTURAL_ALPHA * Math.log1p(descFileScore)
    return descFileScore
  }

  for (const root of roots) dfs(root)
}

function computeDP(node: BuildNode, budget: number, memo: Map<number, NodeDP>): NodeDP {
  if (memo.has(node.id)) return memo.get(node.id)!

  if (node.kind === 'file') {
    const maxCost = node.costTokens
    const len = Math.min(maxCost, budget) + 1
    const dp = new Float64Array(len)
    if (node.costTokens < len) dp[node.costTokens] = node.score
    const info = { dp, maxCost }
    memo.set(node.id, info)
    return info
  }

  const childInfos = node.children.map(child => computeDP(child, budget, memo))
  const childMax = childInfos.reduce((sum, info) => sum + info.maxCost, 0)
  const maxCost = node.costTokens + childMax
  const bound = Math.min(maxCost, budget)

  let merged: ReturnType<typeof merge> = new Float64Array(1) as ReturnType<typeof merge>
  merged[0] = 0
  for (const info of childInfos) {
    merged = merge(merged, info.dp, budget)
  }

  const dp = new Float64Array(bound + 1)
  for (let cb = 0; cb < merged.length; cb++) {
    const b = cb + node.costTokens
    if (b > bound) continue
    const v = merged[cb] + node.score
    if (v > dp[b]) dp[b] = v
  }

  const info = { dp, maxCost }
  memo.set(node.id, info)
  return info
}

function approx(a: number, b: number): boolean {
  return Math.abs(a - b) <= EPSILON
}

function reconstructDirChildrenBudgets(
  node: BuildNode,
  totalBudget: number,
  childDps: Float64Array[],
  budgetLimit: number
): number[] {
  if (node.children.length === 0) return []

  const prefix: Float64Array[] = [new Float64Array(1)]
  prefix[0][0] = 0
  for (let i = 0; i < childDps.length; i++) {
    prefix.push(merge(prefix[i], childDps[i], budgetLimit))
  }

  const split = new Array(node.children.length).fill(0)
  let remain = totalBudget

  for (let i = node.children.length - 1; i >= 0; i--) {
    const prev = prefix[i]
    const full = prefix[i + 1]
    let found = 0
    for (let b = 0; b < childDps[i].length && b <= remain; b++) {
      const left = remain - b
      if (left >= prev.length) continue
      const target = full[remain]
      const val = prev[left] + childDps[i][b]
      if (approx(val, target)) {
        found = b
        break
      }
    }
    split[i] = found
    remain -= found
  }

  return split
}

function reconstructNode(
  node: BuildNode,
  budget: number,
  budgetLimit: number,
  memo: Map<number, NodeDP>,
  selected: Set<number>
): void {
  const info = memo.get(node.id)
  if (!info || budget <= 0 || budget >= info.dp.length) return

  if (info.dp[budget] <= EPSILON) return

  if (node.kind === 'file') {
    if (budget === node.costTokens && approx(info.dp[budget], node.score)) {
      selected.add(node.id)
    }
    return
  }

  if (budget < node.costTokens) return

  selected.add(node.id)
  const childDps = node.children.map(c => memo.get(c.id)!.dp)
  const childBudgetTotal = budget - node.costTokens
  const splits = reconstructDirChildrenBudgets(node, childBudgetTotal, childDps, budgetLimit)

  for (let i = 0; i < node.children.length; i++) {
    const b = splits[i]
    if (b > 0) reconstructNode(node.children[i], b, budgetLimit, memo, selected)
  }
}

function render(roots: BuildNode[], selected: Set<number>): string {
  const lines: string[] = []

  function renderNode(node: BuildNode): void {
    if (!selected.has(node.id)) return
    const indent = '  '.repeat(node.depth)
    lines.push(`${indent}${node.kind === 'dir' ? `${node.name}/` : node.name}`)

    if (node.kind !== 'dir') return
    const visible = node.children.filter(c => selected.has(c.id))
    for (const child of visible) renderNode(child)

    const hiddenCount = node.children.length - visible.length
    if (hiddenCount > 0) {
      lines.push(`${'  '.repeat(node.depth + 1)}... (${hiddenCount} more)`)
    }
  }

  const visibleRoots = roots.filter(r => selected.has(r.id))
  for (const root of visibleRoots) renderNode(root)

  const hiddenRoots = roots.length - visibleRoots.length
  if (hiddenRoots > 0) {
    lines.push(`... (${hiddenRoots} more)`)
  }

  return lines.join('\n')
}

export async function knapsackFolderTree(cwd: string, budgetTokens = 2500): Promise<string> {
  if (budgetTokens <= 0) return ''

  const ignore = createDefaultIgnore()
  const entries = await walk(cwd, cwd, 0, undefined, ignore, {
    respectGitignore: true,
    collectSizes: true,
    collectMtimes: true
  })

  const nonHidden = entries.filter(e => !e.name.startsWith('.'))
  if (nonHidden.length === 0) return ''

  const roots = buildNodeTree(nonHidden)
  if (roots.length === 0) return ''

  const currentDays = nowDays()
  const gitRaw = await collectGitRawScores(cwd, currentDays)
  applyScores(roots, nonHidden, gitRaw, currentDays)

  const memo = new Map<number, NodeDP>()
  const rootDps = roots.map(root => computeDP(root, budgetTokens, memo).dp)

  const prefix: Float64Array[] = [new Float64Array(1)]
  prefix[0][0] = 0
  for (let i = 0; i < rootDps.length; i++) {
    prefix.push(merge(prefix[i], rootDps[i], budgetTokens))
  }

  const all = prefix[prefix.length - 1]
  let bestBudget = 0
  let bestScore = 0
  for (let b = 0; b < all.length; b++) {
    if (all[b] > bestScore + EPSILON) {
      bestScore = all[b]
      bestBudget = b
    }
  }

  if (bestScore <= EPSILON) {
    return ''
  }

  const selected = new Set<number>()
  let remain = bestBudget
  for (let i = roots.length - 1; i >= 0; i--) {
    const prev = prefix[i]
    const full = prefix[i + 1]
    let picked = 0
    for (let b = 0; b < rootDps[i].length && b <= remain; b++) {
      const left = remain - b
      if (left >= prev.length) continue
      if (approx(prev[left] + rootDps[i][b], full[remain])) {
        picked = b
        break
      }
    }
    if (picked > 0) reconstructNode(roots[i], picked, budgetTokens, memo, selected)
    remain -= picked
  }

  return render(roots, selected)
}