import type { ToolResult as EngineToolResult } from '@magnitudedev/turn-engine'
import type { ToolResult } from '../events'

/**
 * Maps the engine's internal ToolResult to the app-level ToolResult.
 * Used by both ExecutionManager (live tool completion) and interrupt
 * recovery (reconstructing tool results from replay state).
 */
export function mapEngineToolResult(result: EngineToolResult): ToolResult {
  switch (result._tag) {
    case 'Success':
      return { status: 'success', output: result.output }
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

/**
 * Legacy alias used by ExecutionManager (xml-act path) — same shape as engine
 * result. Kept for backward compatibility while xml-act execution path is alive.
 */
export const mapXmlToolResult = mapEngineToolResult