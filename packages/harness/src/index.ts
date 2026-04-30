// @magnitudedev/harness

// Streaming partial (self-contained, no tools dependency)
export type { StreamingLeaf, StreamingPartial, DeepPaths } from "./tool/streaming-partial"
export { applyFieldChunk, extractStreamingPartialValues } from "./tool/streaming-partial"

// State model
export type { Phase, BaseState, StateModel } from "./tool/state-model"
export { defineStateModel } from "./tool/state-model"

// Tool
export type { HarnessTool, HarnessToolErased, HarnessToolConcrete, ToolContext, StreamHook } from "./tool/tool"
export { defineHarnessTool } from "./tool/tool"

// Toolkit
export type { ToolkitEntry, Toolkit, ToolkitKeys, ToolkitTool, ToolkitState, ToolRequirements, ToolkitRequirements } from "./tool/toolkit"
export { defineToolkit, mergeToolkits } from "./tool/toolkit"

// Tool handle
export type { ToolHandle } from "./tool/tool-handle"
export { createToolHandle } from "./tool/tool-handle"

// Events
export type {
  ToolResult,
  SafetyStopReason,
  TurnOutcome,
  ToolInputStarted,
  ToolInputFieldChunk,
  ToolInputFieldComplete,
  ToolInputReady,
  ToolExecutionStarted,
  ToolExecutionEnded,
  ToolEmission,
  ToolResultFormatted,
  ThoughtStart,
  ThoughtDelta,
  ThoughtEnd,
  MessageStart,
  MessageDelta,
  MessageEnd,
  ToolInputDecodeFailure,
  TurnStructureDecodeFailure,
  TurnEnd,
  ToolLifecycleEvent,
  HarnessEvent,
} from "./events"

// Hooks
export type { ExecuteHookContext, InterceptorDecision, HarnessHooks } from "./hooks"

// Reducers
export type { Reducer, CanonicalTurnState, CanonicalAccumulator, EngineState, ToolOutcome, ToolHandleState } from "./turn/reducers"
export { CanonicalAccumulatorReducer, EngineStateReducer, createToolHandleReducer, projectCanonical } from "./turn/reducers"

// Dispatcher
export type { DispatchConfig } from "./turn/dispatcher"
export { dispatch } from "./turn/dispatcher"

// Result formation
export { formatToolResult } from "./turn/result-formation"

// Harness
export type { HarnessConfig, Harness, LiveTurn, ReplayTurn } from "./turn/harness"
export { createHarness } from "./turn/harness"
