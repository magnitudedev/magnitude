/**
 * @magnitudedev/tools — Tool Contract Package
 *
 * Pure contract for defining tools. Depends only on Effect.
 * Knows nothing about sandboxes, groups, agents, or runtimes.
 */

export { createTool, Tool } from './tool'
export type { ToolConfig } from './tool'

export type {
  ToolBindings,
  OpenAIBinding,
  CodegenBinding,
  XmlBinding,
  XmlChildBinding,
  XmlChildTagBinding,
  XmlArrayChildBinding,
  CustomToolFormat,
  InputFields,
  FieldPath,
  ArrayFields,
  ArrayElement,
  ChildTagPath,
  AttrNames,
  ChildTagNames,
  DeriveStreamingShape,
} from './bindings'

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
} from './ets'
export type { ToolInterfaceResult, ToolInterfaceOptions, ToolGroupInterfaceOptions, AstGenOptions, KnownEntityInfo } from './ets'

// New tool system contracts
export { defineTool, type ToolDefinition, type ToolDefinitionConfig, type AnyTool, type AnyToolDefinition } from './tool-definition'
export * from './tool-context'
export * from './tool-state-event'
export * from './tool-binding'
export * from './streaming-input'
export * from './state-model'
export {
  defineDisplay,
  createBinding,
  type Display,
  type DisplayConfig,
  type DisplayProps,
  type ToolDisplayBinding,
  type ToolResult,
  type CallState,
  type BindingState,
  type BindingStreaming,
  type BindingInput,
  type BindingOutput,
  type BindingEmission,
} from './display'
export * from './composition'
export * from './models'

export * from './registry'

