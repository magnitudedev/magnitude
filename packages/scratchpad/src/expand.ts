import { resolve, relative, sep } from 'node:path'

/**
 * Result of expanding a $M scratchpad reference in a path string.
 */
export interface ExpandedScratchpadPath {
  /** The resolved absolute path (or original path if not expanded). */
  path: string
  /** Whether $M expansion was performed. */
  expanded: boolean
  /** Display-friendly path relative to scratchpad. Same as `path` when expanded=false. */
  displayPath: string
}

/**
 * Expand $M scratchpad references in a path string.
 *
 * Handles these patterns:
 *   $M           → scratchpadPath
 *   ${M}         → scratchpadPath
 *   $M/foo       → <scratchpadPath>/foo
 *   ${M}/foo     → <scratchpadPath>/foo
 *   ./$M/foo     → <scratchpadPath>/foo  (strip leading ./)
 *   ../$M/foo    → <scratchpadPath>/foo  (strip leading ../)
 *
 * Security model: after detecting the $M token, we resolve the inner path
 * with path.resolve() and then verify the resolved absolute path stays within
 * the scratchpad root. If it escapes, the original input is returned unchanged.
 *
 * Non-$M paths are returned unchanged, preserving any ./ or ../ prefix
 * for the caller to resolve relative to cwd.
 */
export function expandScratchpadPath(inputPath: string, scratchpadPath: string): ExpandedScratchpadPath {
  const notExpanded: ExpandedScratchpadPath = { path: inputPath, expanded: false, displayPath: inputPath }

  // 1. Empty string
  if (inputPath === '') return notExpanded

  // 2. Strip leading dot-segments to find $M token
  let s = inputPath
  while (s.startsWith('./')) s = s.slice(2)
  while (s.startsWith('../')) s = s.slice(3)

  // 3. Exact match: $M or ${M}
  if (s === '$M' || s === '${M}') {
    return { path: scratchpadPath, expanded: true, displayPath: '' }
  }

  // 4. Prefix match: $M/... or ${M}/...
  let innerPath: string | null = null
  if (s.startsWith('$M/')) {
    innerPath = s.slice('$M/'.length)
  } else if (s.startsWith('${M}/')) {
    innerPath = s.slice('${M}/'.length)
  }

  if (innerPath === null) return notExpanded

  // 5-6. Resolve: let path.resolve handle ALL normalization
  const resolved = resolve(scratchpadPath, innerPath)

  // 7. Boundary check: resolved path must be within scratchpad root
  if (resolved === scratchpadPath || resolved.startsWith(scratchpadPath + sep)) {
    // 8. displayPath: relative path from scratchpad root
    const displayPath = relative(scratchpadPath, resolved)
    return { path: resolved, expanded: true, displayPath }
  }

  // Escape — return original unchanged
  return notExpanded
}
