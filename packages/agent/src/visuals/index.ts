/**
 * Visual Reducers — Barrel Export
 *
 * All tool visual reducers and their state types, plus the registry singleton.
 */

// Registry
export { setVisualRegistry, getVisualRegistry } from './registry'
export type { ToolVisualReducer, VisualReducerRegistry } from './registry'

// Define helpers
export { defineToolReducer, reducer, defineCluster } from './define'
export type { ToolReducerConfig, SimpleReducerConfig, ClusterFactory } from './define'

// Shell
export { shellReducer } from './shell'
export type { ShellState } from './shell'

// Filesystem
export { readReducer, writeReducer, editReducer, treeReducer, searchReducer } from './fs'
export type { ReadState, WriteState, EditState, TreeState, TreeEntry, SearchState, SearchMatch } from './fs'

// All other tools
export {
  webSearchReducer, webFetchReducer,

  clickReducer,
  doubleClickReducer,
  rightClickReducer,
  typeReducer,
  scrollReducer,
  dragReducer,
  navigateReducer,
  goBackReducer,
  switchTabReducer,
  newTabReducer,
  screenshotReducer,
  evaluateReducer,
  artifactSyncReducer,
  artifactReadReducer,
  artifactWriteReducer,
  artifactUpdateReducer,
  agentCreateReducer,
  agentPauseReducer,
  agentDismissReducer,
  agentMessageReducer,
  parentMessageReducer,

  skillReducer,
  // Shared helpers/types
  resolveEndPhase,
  isActive,
} from './tools'

export type {
  Phase,
  WebSearchState, WebFetchState,

  BrowserState,
  ArtifactVisualState,
  ArtifactSyncState,
  AgentCreateState,
  AgentIdState,
  AgentMessageState,
  ParentMessageState,

  SkillState,
} from './tools'

