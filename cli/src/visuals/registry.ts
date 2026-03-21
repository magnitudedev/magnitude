/**
 * Visual Registry
 *
 * Display binding registry (CLI) — maps tool keys to model/display bindings.
 */


import { defaultModel, createBinding, type DisplayBindingRegistry, emptyStreamingInput } from '@magnitudedev/tools'
import {
  diffModel, contentModel, shellModel,
  fileReadModel, fileTreeModel, fileSearchModel, webSearchModel, webFetchModel,
  agentCreateModel, agentKillModel,
  skillModel, browserActionModel,
} from '@magnitudedev/agent/src/models'
import {
  defaultDisplay, diffDisplay, contentDisplay, shellDisplay,
  fileReadDisplay, fileTreeDisplay, fileSearchDisplay, webSearchDisplay, webFetchDisplay,
  agentCreateDisplay, agentKillDisplay,
  skillDisplay, browserActionDisplay,
} from './displays'

// === Tool display binding composition root ===

// Display binding composition root.
// Uses createBinding(model, display, initialStreaming) which verifies model↔display type compatibility.
// Full chain verification (tool→xml→model→display) via composeToolChain() is done
// at the agent package level where tool definitions and XML bindings are available.
const newBindings = {
  default: createBinding(defaultModel, defaultDisplay, emptyStreamingInput()),
  diff: createBinding(diffModel, diffDisplay, emptyStreamingInput()),
  content: createBinding(contentModel, contentDisplay, emptyStreamingInput()),
  shell: createBinding(shellModel, shellDisplay, emptyStreamingInput()),
  fileRead: createBinding(fileReadModel, fileReadDisplay, emptyStreamingInput()),
  fileTree: createBinding(fileTreeModel, fileTreeDisplay, emptyStreamingInput()),
  fileSearch: createBinding(fileSearchModel, fileSearchDisplay, emptyStreamingInput()),
  webSearch: createBinding(webSearchModel, webSearchDisplay, emptyStreamingInput()),
  webFetch: createBinding(webFetchModel, webFetchDisplay, emptyStreamingInput()),
  agentCreate: createBinding(agentCreateModel, agentCreateDisplay, emptyStreamingInput()),
  agentKill: createBinding(agentKillModel, agentKillDisplay, emptyStreamingInput()),
  skill: createBinding(skillModel, skillDisplay, emptyStreamingInput()),
  browserAction: createBinding(browserActionModel, browserActionDisplay, emptyStreamingInput()),
} as const;

type CliDisplayContracts = typeof newBindings;

export const displayBindingRegistry: DisplayBindingRegistry<CliDisplayContracts> = {
  get(toolKey) {
    if (
      toolKey === 'default' || toolKey === 'diff' || toolKey === 'content' || toolKey === 'shell'
      || toolKey === 'fileRead' || toolKey === 'fileTree' || toolKey === 'fileSearch'
      || toolKey === 'webSearch' || toolKey === 'webFetch'
      || toolKey === 'agentCreate' || toolKey === 'agentKill'
      || toolKey === 'skill' || toolKey === 'browserAction'
    ) {
      return newBindings[toolKey];
    }
    return undefined;
  },
  getAny(toolKey: string) {
    switch (toolKey) {
      case 'fileEdit': return newBindings.diff;
      case 'fileWrite': return newBindings.content;
      case 'shell': return newBindings.shell;
      case 'fileRead': return newBindings.fileRead;
      case 'fileTree': return newBindings.fileTree;
      case 'fileSearch': return newBindings.fileSearch;
      case 'webSearch': return newBindings.webSearch;
      case 'webFetch': return newBindings.webFetch;
      case 'agentCreate': return newBindings.agentCreate;
      case 'agentKill': return newBindings.agentKill;
      case 'skill': return newBindings.skill;
      case 'click': case 'doubleClick': case 'rightClick': case 'type':
      case 'scroll': case 'drag': case 'navigate': case 'goBack':
      case 'switchTab': case 'newTab': case 'screenshot': case 'evaluate':
        return newBindings.browserAction;
      default: return newBindings.default;
    }
  },
  getDefault() { return newBindings.default; },
};
