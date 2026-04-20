/**
 * @magnitudedev/xml-act — Format Runtime
 *
 * Parses streaming format output and dispatches tool calls via Effect.
 */

// Runtime
export { createRuntime } from './runtime'
export type { Runtime } from './runtime'

// Reactor state
export { initialReactorState, foldReactorState } from './execution/reactor-state'

// Core types — events
export type {
  RuntimeEvent,
  ToolInputStarted,
  ToolInputFieldValue,
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
  LensStart,
  LensChunk,
  LensEnd,
  MessageStart,
  MessageChunk,
  MessageEnd,
  TurnControl,
} from './types'

// Core types — tokens and parse events
export type {
  Token,
  ParseEvent,
  ParameterStarted,
  ParameterChunk,
  ParameterComplete,
  FilterStarted,
  FilterChunk,
  FilterComplete,
  InvokeStarted,
  InvokeComplete,
  StructuralEvent,
} from './types'

// Core types — errors
export type { ParseErrorDetail, StructuralParseErrorDetail } from './types'
export type { ToolCallContext } from './types'

// Core types — results
export type {
  ToolResult,
  ExecutionResult,
} from './types'

// Core types — configuration
export type {
  RuntimeConfig,
  RegisteredTool,
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
  TurnEngineCrash,
} from './types'

// Parameter schema derivation
export { deriveParameters } from './execution/parameter-schema'
export type { ParameterSchema, ToolSchema, ScalarType } from './execution/parameter-schema'

// Input building
export { buildInput, coerceParameterValue } from './execution/input-builder'
export type { ParsedParameter, ParsedInvoke } from './execution/input-builder'

// Output query (JSONPath-based)
export {
  queryOutput,
  renderFilteredResult,
  renderResultBlock,
  observeOutput,
  QueryPatterns,
} from './output-query'
export type { QueryResult } from './output-query'

// Output renderer
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
} from './output-renderer'
export type { RenderConfig } from './output-renderer'

// Result persistence
export {
  getResultsDir,
  ensureResultsDir,
  getResultPath,
  persistResult,
  loadResult,
  loadResultFromPath,
  hasResult,
  deleteResult,
  listResults,
  cleanupResults,
} from './result-persistence'

// Constants
export {
  YIELD_USER_TARGET,
  YIELD_TOOL_TARGET,
  YIELD_WORKER_TARGET,
  YIELD_PARENT_TARGET,
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

// Tokenizer
export { createTokenizer } from './tokenizer'
export type { Tokenizer } from './tokenizer'

// Parser
export { createParser } from './parser'
export type { Parser, ParserEvent, Frame } from './parser'

// Machine
export { createStackMachine } from './machine'
export type { Op, StackMachine } from './machine'

// Grammar builder
export {
  GrammarBuilder,
  type GrammarToolDef,
  type GrammarParameterDef,
  type ProtocolConfig,
  type GrammarConfig,
  type GrammarBuildOptions,
} from './grammar-builder'

// JSONish streaming parser
export { createStreamingJsonParser } from './jsonish/parser'
export type { StreamingJsonParser, ParsedValue, CompletionState } from './jsonish/types'

// JSONish schema coercer
export { coerceToStreamingPartial, tryCastToStreamingPartial } from './jsonish/coercer'
export type { CoercionFlag, CoercedResult } from './jsonish/coercer'

// JSONish parameter accumulator
export { createParameterAccumulator } from './jsonish/accumulator'
