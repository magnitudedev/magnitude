/**
 * @magnitudedev/xml-act — Format Runtime
 *
 * Parses streaming format output and dispatches tool calls via Effect.
 */

// Runtime
export { createTurnEngine } from './engine/turn-engine'
export type { TurnEngine, TurnEngineConfig } from './engine/turn-engine'

// Engine state
export { initialEngineState, foldEngineState } from './engine/engine-state'

// Core types — events
export type {
  TurnEngineEvent,
  ToolLifecycleEvent,
  ToolInputStarted,
  ToolInputFieldChunk,
  ToolInputFieldComplete,
  ToolInputReady,
  ToolParseErrorEvent,
  StructuralParseErrorEvent,
  ProseChunk,
  ProseEnd,
  ToolExecutionStarted,
  ToolExecutionEnded,
  ToolEmission,
  ToolObservation,
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
export type {
  ParseErrorDetail,
  ToolParseError,
  StructuralParseError,
  UnknownToolError,
  UnknownParameterError,
  IncompleteToolError,
  JsonStructuralError,
  SchemaCoercionError,
  MissingRequiredFieldError,
  MalformedTagError,
  StrayCloseTagError,
  MissingToolNameError,
  UnexpectedContentError,
  UnclosedThinkError,
} from './types'

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

// Core types — engine state
export type {
  EngineState,
  ToolOutcome,
} from './types'

// Generic path and streaming types
export type {
  DeepPaths,
  StreamingPartial,
  ApplyFieldChunk,
} from './types'

// Service tags
export {
  ToolInterceptorTag,
  TurnEngineCrash,
} from './types'

// Parameter schema derivation
export { deriveParameters } from './engine/parameter-schema'
export type { ParameterSchema, ToolSchema, ScalarType } from './engine/parameter-schema'

// Output (query, rendering, persistence)
export {
  queryOutput,
  renderFilteredResult,
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
  getResultPath,
  persistResult,
  loadResult,
  loadResultFromPath,
  hasResult,
  deleteResult,
  listResults,
  cleanupResults,
} from './output'
export type { QueryResult, RenderConfig } from './output'

// Constants
export {
  YIELD_USER_TARGET,
  YIELD_INVOKE_TARGET,
  YIELD_WORKER_TARGET,
  YIELD_PARENT_TARGET,
  YIELD_USER,
  YIELD_INVOKE,
  YIELD_WORKER,
  YIELD_PARENT,
  YIELD_USER_STOP,
  YIELD_INVOKE_STOP,
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

// Parser (emits TurnEngineEvent directly, integrates jsonish)
export { createParser } from './parser/index'
export type { XmlActParser, ParserConfig } from './parser/index'

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
} from './grammar/grammar-builder'

// JSONish streaming parser
export { createStreamingJsonParser } from './jsonish/parser'
export type { StreamingJsonParser, ParsedValue, CompletionState } from './jsonish/types'

// JSONish schema coercer
export { coerceToStreamingPartial, tryCastToStreamingPartial } from './jsonish/coercer'
export type { CoercionFlag, CoercedResult } from './jsonish/coercer'


