/**
 * Visual Registries
 *
 * 1. Render registry (CLI) — maps toolKey → render function for think-block rendering.
 * 2. Reducer registry (agent) — set via setVisualRegistry() so DisplayProjection
 *    can reduce visual state as events stream in.
 */

import { createRenderRegistry, createClusterRenderRegistry, createLiveTextRegistry } from './define'
import {
  setVisualRegistry,
  shellReducer,
  readReducer, writeReducer, editReducer, treeReducer, searchReducer,
  webSearchReducer, webFetchReducer,
  clickReducer, doubleClickReducer, rightClickReducer, typeReducer, scrollReducer, dragReducer,
  navigateReducer, goBackReducer, switchTabReducer, newTabReducer, screenshotReducer, evaluateReducer,

  agentCreateReducer, agentDismissReducer, agentMessageReducer, parentMessageReducer,
  skillReducer,
} from '@magnitudedev/agent'
import type { VisualReducerRegistry, ToolVisualReducer } from '@magnitudedev/agent'

// Renderers
import { shellRender, shellLiveText } from './shell'
import { readRender, treeRender, searchRender, readLiveText, treeLiveText, searchLiveText } from './fs'
import {
  webSearchRender, webFetchRender,
  clickRender, doubleClickRender, rightClickRender, typeRender, scrollRender, dragRender,
  navigateRender, goBackRender, switchTabRender, newTabRender, screenshotRender, evaluateRender,
  fsWriteRender, editStreamRender, fsWriteLiveText, editStreamLiveText,
  agentCreateRender, agentDismissRender, agentMessageRender, parentMessageRender,
  skillRender,
  webSearchLiveText, webFetchLiveText, browserLiveText,
  agentCreateLiveText, agentDismissLiveText, agentMessageLiveText, parentMessageLiveText,
  skillLiveText,
} from './tools'

// =============================================================================
// Render registry — toolKey → render function
// =============================================================================

export const renderRegistry = createRenderRegistry({
  shell: shellRender,
  fileRead: readRender,
  fileWrite: fsWriteRender,
  fileEdit: editStreamRender,
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

  agentCreate: agentCreateRender,

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

  agentCreateReducer, agentDismissReducer, agentMessageReducer, parentMessageReducer,
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

export const liveTextRegistry = createLiveTextRegistry({
  shell: ({ state }) => shellLiveText({ state: state as any }),
  fileRead: ({ state }) => readLiveText({ state: state as any }),
  fileWrite: ({ state }) => fsWriteLiveText({ state: state as any }),
  fileEdit: ({ state }) => editStreamLiveText({ state: state as any }),
  fileTree: ({ state }) => treeLiveText({ state: state as any }),
  fileSearch: ({ state }) => searchLiveText({ state: state as any }),
  webSearch: ({ state }) => webSearchLiveText({ state: state as any }),
  webFetch: ({ state }) => webFetchLiveText({ state: state as any }),

  click: ({ state }) => browserLiveText({ state: state as any }),
  doubleClick: ({ state }) => browserLiveText({ state: state as any }),
  rightClick: ({ state }) => browserLiveText({ state: state as any }),
  type: ({ state }) => browserLiveText({ state: state as any }),
  scroll: ({ state }) => browserLiveText({ state: state as any }),
  drag: ({ state }) => browserLiveText({ state: state as any }),
  navigate: ({ state }) => browserLiveText({ state: state as any }),
  goBack: ({ state }) => browserLiveText({ state: state as any }),
  switchTab: ({ state }) => browserLiveText({ state: state as any }),
  newTab: ({ state }) => browserLiveText({ state: state as any }),
  screenshot: ({ state }) => browserLiveText({ state: state as any }),
  evaluate: ({ state }) => browserLiveText({ state: state as any }),

  agentCreate: ({ state }) => agentCreateLiveText({ state: state as any }),

  agentDismiss: ({ state }) => agentDismissLiveText({ state: state as any }),
  agentMessage: ({ state }) => agentMessageLiveText({ state: state as any }),
  parentMessage: ({ state }) => parentMessageLiveText({ state: state as any }),
  skill: ({ state }) => skillLiveText({ state: state as any }),
})

export const clusterRenderRegistry = createClusterRenderRegistry({
})
