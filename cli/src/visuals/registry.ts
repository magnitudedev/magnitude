/**
 * Visual Registries
 *
 * 1. Render registry (CLI) — maps toolKey → render function for think-block rendering.
 * 2. Reducer registry (agent) — set via setVisualRegistry() so DisplayProjection
 *    can reduce visual state as events stream in.
 */

import { createRenderRegistry, createClusterRenderRegistry } from './define'
import {
  setVisualRegistry,
  shellReducer,
  readReducer, writeReducer, editReducer, treeReducer, searchReducer,
  webSearchReducer, webFetchReducer,
  clickReducer, doubleClickReducer, rightClickReducer, typeReducer, scrollReducer, dragReducer,
  navigateReducer, goBackReducer, switchTabReducer, newTabReducer, screenshotReducer, evaluateReducer,
  artifactSyncReducer, artifactReadReducer, artifactWriteReducer, artifactUpdateReducer,
  agentCreateReducer, agentPauseReducer, agentDismissReducer, agentMessageReducer, parentMessageReducer,
  skillReducer,
} from '@magnitudedev/agent'
import type { VisualReducerRegistry, ToolVisualReducer } from '@magnitudedev/agent'

// Renderers
import { shellRender } from './shell'
import { readRender, writeRender, editRender, editClusterRender, treeRender, searchRender } from './fs'
import {
  webSearchRender, webFetchRender,
  clickRender, doubleClickRender, rightClickRender, typeRender, scrollRender, dragRender,
  navigateRender, goBackRender, switchTabRender, newTabRender, screenshotRender, evaluateRender,
  artifactSyncRender, artifactReadRender, artifactWriteRender, artifactUpdateRender,
  agentCreateRender, agentPauseRender, agentDismissRender, agentMessageRender, parentMessageRender,
  skillRender,
} from './tools'

// =============================================================================
// Render registry — toolKey → render function
// =============================================================================

export const renderRegistry = createRenderRegistry({
  shell: shellRender,
  fileRead: readRender,
  fileWrite: writeRender,
  fileEdit: editRender,
  fileTree: treeRender,
  fileSearch: searchRender,
  webSearch: webSearchRender,
  webFetch: webFetchRender,

  click: clickRender,
  doubleClick: doubleClickRender,
  rightClick: rightClickRender,
  type: typeRender,
  scroll: scrollRender,
  drag: dragRender,
  navigate: navigateRender,
  goBack: goBackRender,
  switchTab: switchTabRender,
  newTab: newTabRender,
  screenshot: screenshotRender,
  evaluate: evaluateRender,
  artifactSync: artifactSyncRender,
  artifactRead: artifactReadRender,
  artifactWrite: artifactWriteRender,
  artifactUpdate: artifactUpdateRender,
  agentCreate: agentCreateRender,
  agentPause: agentPauseRender,
  agentDismiss: agentDismissRender,
  agentMessage: agentMessageRender,
  parentMessage: parentMessageRender,

  skill: skillRender,
})

// =============================================================================
// Reducer registry — set on agent package for DisplayProjection
// =============================================================================

const allReducers: readonly ToolVisualReducer[] = [
  shellReducer,
  readReducer, writeReducer, editReducer, treeReducer, searchReducer,
  webSearchReducer, webFetchReducer,
  clickReducer, doubleClickReducer, rightClickReducer, typeReducer, scrollReducer, dragReducer,
  navigateReducer, goBackReducer, switchTabReducer, newTabReducer, screenshotReducer, evaluateReducer,
  artifactSyncReducer, artifactReadReducer, artifactWriteReducer, artifactUpdateReducer,
  agentCreateReducer, agentPauseReducer, agentDismissReducer, agentMessageReducer, parentMessageReducer,
  skillReducer,
]

const reducerMap = new Map<string, ToolVisualReducer>(
  allReducers.map(r => [r.toolKey, r])
)
const reducerRegistry: VisualReducerRegistry = {
  get: (toolKey: string) => reducerMap.get(toolKey),
}
setVisualRegistry(reducerRegistry)

// =============================================================================
// Cluster render registry — cluster key → cluster render function
// =============================================================================

export const clusterRenderRegistry = createClusterRenderRegistry({
  edit: editClusterRender,
})
