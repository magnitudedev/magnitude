/**
 * @magnitudedev/turn-engine
 *
 * Format-neutral turn execution engine. Consumes a stream of
 * ResponseStreamEvent values from the codec, translates them into engine
 * events, drives tool dispatch, tracks replay, and emits an outcome.
 */

export type {
  // Source spans
  SourcePos,
  SourceSpan,

  // Tool registration
  RegisteredTool,

  // Reasoning / messaging
  ThoughtStart,
  ThoughtChunk,
  ThoughtEnd,
  MessageStart,
  MessageChunk,
  MessageEnd,

  // Tool input lifecycle
  ToolCallContext,
  ToolInputStarted,
  ToolInputFieldChunk,
  ToolInputFieldComplete,
  ToolInputReady,

  // Tool execution lifecycle
  ToolExecutionStarted,
  ToolExecutionEnded,
  ToolEmission,

  // Decode failures
  ToolInputDecodeFailure,
  TurnStructureDecodeFailure,

  // Turn end / outcome
  TurnControl,
  TurnEnd,
  TurnEngineOutcome,
  SafetyStopReason,

  // Result / outcome types
  ToolResult,
  ToolOutcome,

  // Engine state
  EngineState,

  // Interceptor
  InterceptorContext,
  InterceptorDecision,
  ToolInterceptor,

  // Narrowed lifecycle view
  ToolLifecycleEvent,
  ApplyFieldChunk,

  // Top-level event union
  TurnEngineEvent,

  // Runtime config
  RuntimeConfig,

  // Re-exports from @magnitudedev/tools (convenience)
  ContentPart,
  DeepPaths,
  StreamingPartial,
} from './types'

export { TurnEngineCrash, ToolInterceptorTag } from './types'

// =============================================================================
// Engine
// =============================================================================
export { createTurnEngine } from './turn-engine'
export type { TurnEngine, TurnEngineConfig } from './turn-engine'

// =============================================================================
// Engine state (folded over event stream)
// =============================================================================
export { initialEngineState, foldEngineState } from './engine-state'

// =============================================================================
// Dispatcher (low-level — engine uses this internally)
// =============================================================================
export { dispatchTool } from './dispatcher'
export type { DispatchContext, DispatchInput, DispatchResult } from './dispatcher'

// =============================================================================
// Input builder (for state-model consumers building streaming input partials)
// =============================================================================
export { applyFieldChunk } from './input-builder'

// =============================================================================
// Tool output renderer (pure function — call when building memory entries)
// =============================================================================
export { renderToolOutput } from './render-output'


