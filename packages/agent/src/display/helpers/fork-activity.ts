import type { ForkActivityToolCounts } from '../types'
import type { ToolKey } from '../../tools/toolkits'

export function incrementToolCount(counts: ForkActivityToolCounts, toolKey: ToolKey): ForkActivityToolCounts {
  switch (toolKey) {
    case 'shell': return { ...counts, commands: counts.commands + 1 }
    case 'fileRead':
    case 'fileTree': return { ...counts, reads: counts.reads + 1 }
    case 'fileWrite': return { ...counts, writes: counts.writes + 1 }
    case 'fileEdit': return { ...counts, edits: counts.edits + 1 }
    case 'fileSearch': return { ...counts, searches: counts.searches + 1 }
    case 'webFetch': return { ...counts, webFetches: counts.webFetches + 1 }
    case 'webSearch': return { ...counts, webSearches: counts.webSearches + 1 }
    case 'fileView':
    case 'queryImage':
      return { ...counts, other: counts.other + 1 }
    default:
      return { ...counts, other: counts.other + 1 }
  }
}

export function totalToolsUsed(counts: ForkActivityToolCounts): number {
  return counts.commands
    + counts.reads
    + counts.writes
    + counts.edits
    + counts.searches
    + counts.webSearches
    + counts.webFetches
    + counts.artifactWrites
    + counts.artifactUpdates
    + counts.other
}
