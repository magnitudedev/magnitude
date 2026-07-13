// @magnitudedev/harness

// Streaming partial (self-contained, no tools dependency)
export type { StreamingLeaf, StreamingPartial, DeepPaths } from "./tool/streaming-partial"
export { applyFieldChunk, extractStreamingPartialValues } from "./tool/streaming-partial"

// State model
export type { Phase, BaseState, StateModel } from "./tool/state-model"
export { BaseStateSchema, PhaseSchema, defineStateModel } from "./tool/state-model"

// Tool
export type { HarnessTool, ToolContext, StreamHook } from "./tool/tool"
export { StreamValidationError } from "./tool/tool"
export { defineHarnessTool } from "./tool/tool"

// Toolkit
export type { ToolkitEntry, Toolkit, ToolkitKeys, ToolkitTool, ToolkitState, ToolRequirements, ToolkitRequirements } from "./tool/toolkit"
export { defineToolkit, mergeToolkits } from "./tool/toolkit"

// Tool handle
export type { ToolHandle } from "./tool/tool-handle"
export { createToolHandle, interruptToolHandle, processToolHandle } from "./tool/tool-handle"

// Events
export type {
  ToolError,
  ToolResult,
  ToolResultEntry,
  SafetyStopReason,
  TurnOutcome,
  ToolInputStarted,
  ToolInputFieldChunk,
  ToolInputFieldComplete,
  ToolInputReady,
  ToolInputRejected,
  ToolExecutionStarted,
  ToolExecutionEnded,
  ToolEmission,
  ThoughtStart,
  ThoughtDelta,
  ThoughtEnd,
  MessageStart,
  MessageDelta,
  MessageEnd,
  TurnEnd,
  ToolLifecycleEvent,
  HarnessEvent,
} from "./events"

// Hooks
export type { ExecuteHookContext, InterceptorDecision, HarnessHooks } from "./hooks"

// Reducers
export type {
  Reducer,
  TurnState,
  TurnStateForToolHandle,
  CanonicalTurnState,
  CanonicalAccumulator,
  EngineState,
  ToolOutcome,
  ToolHandleState,
  ToolHandleStateForHandle,
} from "./turn/reducers"
export {
  CanonicalAccumulatorReducer,
  CanonicalAccumulatorSchema,
  CanonicalTurnStateSchema,
  EngineStateSchema,
  ResponseUsageSchema,
  ToolHandleStateSchema,
  ToolOutcomeSchema,
  ToolResultEntrySchema,
  ToolResultSchema,
  TurnOutcomeSchema,
  TurnStateSchema,
  StreamFailedTerminalSchema,
  createTurnReducer,
  makeKeyedToolHandleSchema,
  makeKeyedToolHandleSchemaEntries,
  makeKeyedToolHandleUnionSchemaFromEntries,
  makeToolHandleSchema,
  makeToolHandleStateSchema,
  makeToolHandleStateSchemaFromHandle,
  makeTurnStateSchema,
  makeTurnStateSchemaFromToolHandle,
  type KeyedToolHandleSchemaEntries,
  projectCanonical,
} from "./turn/reducers"

// Dispatcher
export type { DispatchConfig } from "./turn/dispatcher"
export { dispatch } from "./turn/dispatcher"

// Content building
export { ContentBuilder } from "./content"

// Rendering utilities (used by agent formatting layer)
export { isImageValue, toImagePart, isScalar, renderToolOutput, renderTagged } from "./formatting/helpers"
export { renderSchemaParams } from "@magnitudedev/utils/schema"
export type { ToolResultFormatter } from "./formatting/tool-result-formatter"
export { createToolResultFormatter } from "./formatting/tool-result-formatter"

// Harness
export type { HarnessConfig, Harness, LiveTurn, ReplayTurn } from "./turn/harness"
export { createHarness } from "./turn/harness"
