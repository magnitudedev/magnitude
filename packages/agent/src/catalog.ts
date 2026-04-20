import { defineCatalog } from '@magnitudedev/tools'
import type { StateModel, ToolDefinition, ToolCatalog } from '@magnitudedev/tools'

/** Agent-level catalog entry — every tool in the agent catalog has a state model */
export interface AgentCatalogEntry {
  readonly tool: ToolDefinition
  readonly state: StateModel<any, any, any, any>
  readonly display?: boolean
  readonly group?: string
}

/** Agent catalog — a ToolCatalog whose entries satisfy AgentCatalogEntry */
export type AgentCatalog<T extends Record<string, AgentCatalogEntry> = Record<string, AgentCatalogEntry>> = ToolCatalog<T>

// Tools
import { shellTool } from './tools/shell'
import {
  readTool,
  writeTool,
  editTool,
  treeTool,
  grepTool,
  viewTool,
} from './tools/fs'
import { webFetchTool } from './tools/web-fetch-tool'
import { webSearchTool } from './tools/web-search'

import {
  clickTool,
  doubleClickTool,
  rightClickTool,
  typeTool,
  scrollTool,
  dragTool,
  navigateTool,
  goBackTool,
  switchTabTool,
  newTabTool,
  screenshotTool,
  evaluateTool,
} from './tools/browser-tools'
import {
  createTaskTool,
  updateTaskTool,
  spawnWorkerTool,
  killWorkerTool,
} from './tools/task-tools'
import {
  agentCreateTool,
  agentKillTool,
} from './tools/agent-tools'

// State models
import { shellModel } from './models/shell'
import { fileReadModel } from './models/file-read'
import { fileWriteModel } from './models/file-write'
import { fileEditModel } from './models/file-edit'
import { fileTreeModel } from './models/file-tree'
import { fileSearchModel } from './models/file-search'
import { fileViewModel } from './models/file-view'
import { webFetchModel } from './models/web-fetch'
import { webSearchModel } from './models/web-search'

import {
  clickModel,
  doubleClickModel,
  rightClickModel,
  typeModel,
  scrollModel,
  dragModel,
  navigateModel,
  goBackModel,
  switchTabModel,
  newTabModel,
  screenshotModel,
  evaluateModel,
} from './models/browser-action'
import { createTaskModel } from './models/create-task'
import { updateTaskModel } from './models/update-task'
import { spawnWorkerModel } from './models/spawn-worker'
import { killWorkerModel } from './models/kill-worker'
import { agentCreateModel } from './models/agent-create'
import { agentKillModel } from './models/agent-kill'
import { skillTool } from './tools/skill-tool'
import { skillActivationModel } from './models/skill-activation'

export const catalog = defineCatalog({
  shell: { tool: shellTool, state: shellModel },
  fileRead: { tool: readTool, state: fileReadModel },
  fileWrite: { tool: writeTool, state: fileWriteModel },
  fileEdit: { tool: editTool, state: fileEditModel },
  fileTree: { tool: treeTool, state: fileTreeModel },
  fileSearch: { tool: grepTool, state: fileSearchModel },
  fileView: { tool: viewTool, state: fileViewModel },
  webSearch: { tool: webSearchTool, state: webSearchModel },
  webFetch: { tool: webFetchTool, state: webFetchModel },

  click: { tool: clickTool, state: clickModel, group: 'browser' },
  doubleClick: { tool: doubleClickTool, state: doubleClickModel, group: 'browser' },
  rightClick: { tool: rightClickTool, state: rightClickModel, group: 'browser' },
  type: { tool: typeTool, state: typeModel, group: 'browser' },
  scroll: { tool: scrollTool, state: scrollModel, group: 'browser' },
  drag: { tool: dragTool, state: dragModel, group: 'browser' },
  navigate: { tool: navigateTool, state: navigateModel, group: 'browser' },
  goBack: { tool: goBackTool, state: goBackModel, group: 'browser' },
  switchTab: { tool: switchTabTool, state: switchTabModel, group: 'browser' },
  newTab: { tool: newTabTool, state: newTabModel, group: 'browser' },
  screenshot: { tool: screenshotTool, state: screenshotModel, group: 'browser' },
  evaluate: { tool: evaluateTool, state: evaluateModel, group: 'browser' },

  agentCreate: { tool: agentCreateTool, state: agentCreateModel },
  agentKill: { tool: agentKillTool, state: agentKillModel },

  createTask: { tool: createTaskTool, state: createTaskModel, display: false },
  updateTask: { tool: updateTaskTool, state: updateTaskModel, display: false },
  spawnWorker: { tool: spawnWorkerTool, state: spawnWorkerModel, display: false },
  killWorker: { tool: killWorkerTool, state: killWorkerModel, display: false },
  skill: { tool: skillTool, state: skillActivationModel },
} as const)

export type ToolKey = keyof typeof catalog.entries

export function isToolKey(value: string): value is ToolKey {
  return value in catalog.entries
}
