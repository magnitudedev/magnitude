/**
 * Tools Index
 *
 * Re-exports for the tools package barrel.
 * buildRegisteredTools and generateToolGrammar live in ./tool-registry.
 */

export { getBindingRegistry } from './binding-registry'

// =============================================================================
// Re-exports
// =============================================================================

export { globalTools } from './globals'

export { shellTool } from './shell'
export { fsTools } from './fs'


export { webFetchTool } from './web-fetch-tool'
export { webSearchTool, webSearchXmlBinding } from './web-search'
export {
  createTaskTool,
  updateTaskTool,
  spawnWorkerTool,
  killWorkerTool,
  createTaskXmlBinding,
  updateTaskXmlBinding,
  spawnWorkerXmlBinding,
  killWorkerXmlBinding,
} from './task-tools'
