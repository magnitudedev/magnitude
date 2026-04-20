/**
 * @magnitudedev/tools — Tool Contract Package
 *
 * Pure contract for defining tools. Depends only on Effect.
 * Knows nothing about sandboxes, groups, agents, or runtimes.
 */

export { ToolErrorSchema } from './errors'
export type { ToolErrorBase } from './errors'

export { ToolImageSchema } from './image-types'
export type { ImageMediaType, ToolImageValue, ContentPart } from './image-types'

// ETS: Effect Schema to TypeScript (tool interface generation)
export {
  generateToolInterface,
  generateToolGroupInterface,
  generateNamespaceInterface,
  getTypeNode,
  schemaToTypeNode,
  printAst,
  buildKnownEntities,
  renderToolDocs,
} from './ets'
export type { ToolInterfaceResult, ToolInterfaceOptions, ToolGroupInterfaceOptions, AstGenOptions, KnownEntityInfo } from './ets'

// Tool definition
export { defineTool, type ToolDefinition, type ToolDefinitionConfig, type StreamHook } from './tool-definition'

// Tool context
export * from './tool-context'

// Tool lifecycle events
export type { ToolStateEvent, ToolResult, ParseErrorDetail } from './tool-state-event'

// Streaming partial
export * from './streaming-partial'

// State model
export { defineStateModel } from './state-model'
export type { StateModel, BaseState, Phase } from './state-model'

// Tool call state
export { ToolCallState, createToolCallState } from './tool-call-state'

// Catalog
export { defineCatalog, type ToolCatalog, type PickableCatalog, type BaseCatalogEntry, type CatalogKeys, type CatalogEntry, type CatalogTool } from './tool-catalog'

// Models
export type { EditDiff } from './models/edit-diff'
