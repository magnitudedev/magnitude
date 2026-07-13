/**
 * Tree Building
 * 
 * Convert flat entry lists to tree structures.
 */

export interface FolderNode {
  name: string
  children: FolderNode[]
  lastModified: number
  /** Cumulative size in bytes of all files under this directory */
  totalBytes: number
}

/**
 * Build tree structure from flat list of entries.
 * Accumulates file sizes into parent directories when size data is available.
 * Pure function - no side effects.
 */
export function buildTree<T extends { relativePath: string; name: string; type: string; size?: number }>(
  entries: T[]
): FolderNode[] {
  const nodeMap = new Map<string, FolderNode>()
  const roots: FolderNode[] = []

  // Filter to directories only
  const dirs = entries.filter(e => e.type === 'dir')

  // Create all nodes
  for (const entry of dirs) {
    nodeMap.set(entry.relativePath, {
      name: entry.name,
      children: [],
      lastModified: 0,
      totalBytes: 0,
    })
  }

  // Build hierarchy
  for (const entry of dirs) {
    const node = nodeMap.get(entry.relativePath)!
    const parts = entry.relativePath.split('/')

    if (parts.length === 1) {
      // Root level
      roots.push(node)
    } else {
      // Find parent
      const parentPath = parts.slice(0, -1).join('/')
      const parent = nodeMap.get(parentPath)
      if (parent) {
        parent.children.push(node)
      }
    }
  }

  // Accumulate file sizes into their parent directories
  const files = entries.filter(e => e.type === 'file' && e.size !== undefined)
  for (const file of files) {
    const parts = file.relativePath.split('/')
    if (parts.length < 2) {
      // Root-level file — no parent dir in the tree
      continue
    }
    const parentPath = parts.slice(0, -1).join('/')
    const parent = nodeMap.get(parentPath)
    if (parent) {
      parent.totalBytes += file.size!
    }
  }

  // Bubble up: propagate child totalBytes to parents (post-order)
  function propagate(node: FolderNode): number {
    for (const child of node.children) {
      node.totalBytes += propagate(child)
    }
    return node.totalBytes
  }
  for (const root of roots) {
    propagate(root)
  }

  return roots
}
