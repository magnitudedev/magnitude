
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
export type { ToolCallContext } from './types'
export type {
  TagParseErrorDetail,
  ParseErrorDetail,
  StructuralParseErrorDetail,
  UnclosedThinkDetail,
} from './format/types'

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
export { generateXmlToolDoc, generateXmlToolGroupDoc, generateXmlToolInputShape } from './xml-docs'
export type { XmlToolDocEntry } from './xml-docs'

// Output serializer
export { serializeOutput } from './output-serializer'

// Output query
export { observeOutput } from './output-query'

// Output tree (structured tool output AST)
export { buildOutputTree, outputToText, outputToDOM, outputFromDOM } from './output-tree'
export type { OutputNode } from './output-tree'

// Stream guard
export {
  YIELD_USER_TAG,
  YIELD_TOOL_TAG,
  YIELD_WORKER_TAG,
  YIELD_PARENT_TAG,
  YIELD_USER,
  YIELD_TOOL,
  YIELD_WORKER,
  YIELD_PARENT,
  YIELD_USER_STOP,
  YIELD_TOOL_STOP,
  YIELD_WORKER_STOP,
  YIELD_PARENT_STOP,
  LEAD_YIELD_STOP_SEQUENCES,
  SUBAGENT_YIELD_STOP_SEQUENCES,
  LEAD_YIELD_TAGS,
  SUBAGENT_YIELD_TAGS,
} from './constants'

// Parser
export { createStreamingXmlParser, createParser, defaultIdGenerator } from './parser'
export type { StreamingParser, IdGenerator } from './parser'

// Machine + tokenizer
export { createStackMachine } from './machine'
export type { Op, StackMachine } from './machine'
export { createTokenizer } from './tokenizer'
export type { Tokenizer, Token } from './tokenizer'

// Format
export * from './format'
export type { ParseEvent, ParsedElement, ParsedChild, AttributeValue, XmlActEvent } from './format/types'
export { SchemaAccumulator } from './schema-accumulator'
export { StreamingAccumulator, type StreamingAccumulatorConfig } from './streaming-accumulator'
export { defineXmlBinding, type XmlBindingResult, type XmlMappingConfig, type XmlInputMappingConfig, type XmlOutputBinding } from './xml-binding'
export type { DeriveStreamingShape, DeriveFields, DeriveChildren, AttrNames, ChildTagNames, ChildrenTagNames } from './type-chain'
export * from './constants'

// GBNF grammar for constrained generation (e.g. Fireworks AI)
export { type GrammarToolDef } from './grammar-builder'
export {
  GrammarBuilder,
  type ProtocolConfig,
  type GrammarConfig,
  type GrammarBuildOptions,
} from './grammar-builder'
