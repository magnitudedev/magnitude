import type { XmlToolResult } from '@magnitudedev/xml-act'
import type { ToolDisplay, ToolResult } from '../events'

/**
 * Maps xml-act's internal XmlToolResult to the app-level ToolResult.
 * Used by both ExecutionManager (live tool completion) and interrupt
 * recovery (reconstructing tool results from replay state).
 */
export function mapXmlToolResult(result: XmlToolResult, display?: ToolDisplay): ToolResult {
  switch (result._tag) {
    case 'Success':
      return { status: 'success', output: result.output, ...(display ? { display } : {}) }
    case 'Error':
      return { status: 'error', message: result.error }
    case 'Rejected': {
      const rej = result.rejection
      const isPerm = rej && typeof rej === 'object' && '_tag' in rej
      if (isPerm) {
        const r = rej as { _tag: string; reason: string }
        if (r._tag === 'UserRejection') {
          return { status: 'rejected', message: 'User rejected the action' }
        }
        return { status: 'rejected', message: 'System rejected', reason: r.reason }
      }
      return { status: 'rejected', message: String(rej) }
    }
    case 'Interrupted':
      return { status: 'interrupted' }
  }
}