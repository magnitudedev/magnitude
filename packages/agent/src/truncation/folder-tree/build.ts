import type { FolderNode } from './tree'

/**
 * Build a folder tree from a flat list of folders.
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
