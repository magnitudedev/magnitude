/**
 * Tools Index
 *
 * Re-exports for the tools package barrel.
 * buildRegisteredTools and generateToolGrammar live in ./tool-registry.
 */

export { buildResolvedToolSet, type ResolvedToolSet } from './resolved-toolset'

// =============================================================================
// Re-exports
// =============================================================================

export { globalTools } from './globals'

export { shellTool } from './shell'
export { fsTools } from './fs'

export { webFetchTool } from './web-fetch-tool'
export { webSearchTool } from './web-search'
export {
  createTaskTool,
  updateTaskTool,
  spawnWorkerTool,
  killWorkerTool,
} from './task-tools'
