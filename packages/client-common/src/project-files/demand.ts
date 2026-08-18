import type { ProjectDirectoryEntry, ProjectRelativePath } from "@magnitudedev/sdk"

/**
 * Visit demanded directories breadth-first through authoritative parent entries.
 * Returning no children leaves that branch unresolved until a later evaluation.
 */
export function visitProjectDirectoryDemand(
  rootEntries: readonly ProjectDirectoryEntry[],
  demanded: ReadonlySet<ProjectRelativePath>,
  visit: (directory: ProjectRelativePath) => readonly ProjectDirectoryEntry[] | undefined,
): readonly ProjectRelativePath[] {
  const visited: ProjectRelativePath[] = []
  let level = rootEntries

  while (level.length > 0) {
    const nextLevel: ProjectDirectoryEntry[] = []
    for (const entry of level) {
      if (entry.kind !== "directory" || !demanded.has(entry.path)) continue
      visited.push(entry.path)
      const children = visit(entry.path)
      if (children !== undefined) nextLevel.push(...children)
    }
    level = nextLevel
  }

  return visited
}
