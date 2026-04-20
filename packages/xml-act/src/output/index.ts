/**
 * Output — rendering, querying, and persistence of tool results.
 */

// Query (JSONPath-based filtering)
export { queryOutput, renderFilteredResult } from './query'
export type { QueryResult } from './query'

// Renderer (canonical rendering logic)
export {
  renderResult,
  renderResultBody,
  renderVoidResult,
  renderStringResult,
  renderScalarResult,
  renderArrayResult,
  renderObjectResult,
  renderOutField,
  renderResultToParts,
  renderShellResult,
  renderGrepResult,
  renderReadResult,
  renderWriteResult,
  renderEditResult,
  renderTreeResult,
  renderSkillResult,
  parseResultBlock,
  isValidResultBlock,
  extractToolName,
} from './renderer'
export type { RenderConfig } from './renderer'

// Persistence
export {
  getResultPath,
  persistResult,
  loadResult,
  loadResultFromPath,
  hasResult,
  deleteResult,
  listResults,
  cleanupResults,
} from './persistence'
