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
  ArrayFields,
  ArrayElement,
  ChildTagPath,
} from './bindings'

export { ToolErrorSchema } from './errors'
export type { ToolErrorBase } from './errors'

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

