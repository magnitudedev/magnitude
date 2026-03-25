/**
 * @magnitudedev/xml-act — Standalone XML Tool Execution Runtime
 *
 * Parses streaming XML directly and dispatches tool calls via Effect.
 * No JS generation, no sandbox — just XML parsing → tool dispatch → events.
 */

// Runtime
export { createXmlRuntime } from './execution/xml-runtime'
export type { XmlRuntime } from './execution/xml-runtime'

// Reactor state
export { initialReactorState, foldReactorState } from './execution/reactor-state'

// Core types — events
export type {
  XmlRuntimeEvent,
  ToolCallEvent,
  ToolInputStarted,
  ToolInputFieldValue,
  ToolInputBodyChunk,
  ToolInputChildStarted,
  ToolInputChildComplete,
  ToolInputReady,
  ToolInputParseError,
  ProseChunk,
  ProseEnd,
  ToolExecutionStarted,
  ToolExecutionEnded,
  ToolEmission,
  ToolObservation,
  StructuralParseError,
  TurnEnd,
} from './types'

// Binding type-level helpers
export type {
  ResolvePath,
  BindingAttrs,
  BindingBody,
  BindingChildren,
  BindingChildTags,
  BindingChildRecord,
  ChildBindingField,
  ChildBindingAttrs,
  ChildBindingBody,
  ChildRecordField,
  ChildElem,
  ChildAttrsPick,
  AllBodyFields,
} from './types'

// Core types — errors
export type { ToolCallContext, ToolCallError } from './types'
export type { UnclosedThinkDetail, UnclosedActionsDetail } from './parser/types'

// Core types — results
export type {
  XmlToolResult,
  XmlExecutionResult,
} from './types'

// Core types — configuration
export type {
  XmlRuntimeConfig,
  RegisteredTool,
  XmlTagBinding,
  XmlChildBinding,
} from './types'

// Core types — services
export type {
  ToolInterceptor,
  InterceptorContext,
  InterceptorDecision,
} from './types'

// Core types — reactor
export type {
  ReactorState,
  ToolOutcome,
} from './types'

// Service tags
export {
  ToolInterceptorTag,
  XmlRuntimeCrash,
} from './types'

// Utilities
export { buildInput } from './execution/input-builder'

// Binding validation
export { validateBinding } from './execution/binding-validator'
export type { TagSchema, ChildTagSchema, AttributeSchema } from './execution/binding-validator'

// XML documentation generation
export { generateXmlToolDoc, generateXmlToolGroupDoc, defaultXmlTagName } from './xml-docs'
export type { XmlToolDocEntry } from './xml-docs'

// Output serializer
export { serializeOutput } from './output-serializer'

// Output query
export { observeOutput } from './output-query'

// Output tree (structured tool output AST)
export { buildOutputTree, outputToText, outputToDOM, outputFromDOM } from './output-tree'
export type { OutputNode } from './output-tree'

// Stream guard
export { guardStream, guardEffectStream } from './stream-guard'

// Protocol constants & keyword configuration
export { getKeywords, useAltKeywords, useDefaultKeywords, actionsTagOpen, actionsTagClose, thinkTagOpen, thinkTagClose, commsTagOpen, commsTagClose, TURN_CONTROL_NEXT, TURN_CONTROL_YIELD, TURN_CONTROL_FINISH } from './constants'
export type { KeywordSet } from './constants'

// Parser (for testing — not part of the public API)
export { createStreamingXmlParser, defaultIdGenerator } from './parser/streaming-xml-parser'
export type { StreamingXmlParser, IdGenerator } from './parser/streaming-xml-parser'
export type { ParseEvent, ParsedElement, ParsedChild, AttributeValue } from './parser/types'

export { SchemaAccumulator } from './schema-accumulator'
export { StreamingAccumulator, type StreamingAccumulatorConfig } from './streaming-accumulator'
export { defineXmlBinding, type XmlBindingResult, type XmlMappingConfig, type XmlInputMappingConfig, type XmlOutputBinding } from './xml-binding'
export type { DeriveStreamingShape, DeriveFields, DeriveChildren, AttrNames, ChildTagNames, ChildrenTagNames } from './type-chain'
