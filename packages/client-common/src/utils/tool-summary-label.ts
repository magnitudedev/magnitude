import type { ToolSummaryPresentation } from '@magnitudedev/sdk'

/**
 * Computes the human-readable label for a tool summary entry from its typed
 * aggregate fields. Shared by CLI and web so summary labels do not diverge.
 *
 * The projection owns which tools get grouped and the aggregate counts; the
 * label string is rendering and lives here.
 */
export function toolSummaryLabel(summary: ToolSummaryPresentation): string {
  switch (summary.toolKey) {
    case 'fileRead':
      return `Read ${summary.count} ${summary.count === 1 ? 'file' : 'files'}`
    case 'fileSearch': {
      const matches = summary.matchCount ?? 0
      const files = summary.fileCount ?? 0
      return `${matches} match${matches === 1 ? '' : 'es'} in ${files} file${files === 1 ? '' : 's'}`
    }
    case 'webSearch': {
      const sources = summary.sourceCount ?? 0
      return `${summary.count} web search${summary.count === 1 ? '' : 'es'}, ${sources} total source${sources === 1 ? '' : 's'}`
    }
    case 'webFetch':
      return `Fetch ${summary.count} URL${summary.count === 1 ? '' : 's'}`
    case 'fileTree':
      return 'List files'
    case 'fileView':
      return `View ${summary.count} file${summary.count === 1 ? '' : 's'}`
  }
}
