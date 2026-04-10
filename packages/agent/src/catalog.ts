import { defineCatalog } from '@magnitudedev/tools'
import type { StateModel, ToolDefinition, ToolCatalog } from '@magnitudedev/tools'
import type { XmlTagBinding } from '@magnitudedev/xml-act'

type AgentToolBinding = {
  toXmlTagBinding(): XmlTagBinding
  toXmlOutputBinding(): XmlTagBinding
}

/** Agent-level catalog entry — every tool in the agent catalog has a binding and state model */
export interface AgentCatalogEntry {
  readonly tool: ToolDefinition
  readonly binding: AgentToolBinding
  readonly state: StateModel<any, any, any, any>
  readonly display?: boolean
  readonly group?: string
}

/** Agent catalog — a ToolCatalog whose entries satisfy AgentCatalogEntry */
export type AgentCatalog<T extends Record<string, AgentCatalogEntry> = Record<string, AgentCatalogEntry>> = ToolCatalog<T>

// Tools + Bindings
import { shellTool, shellXmlBinding } from './tools/shell'
import {
  readTool,
  readXmlBinding,
  writeTool,
  writeXmlBinding,
  editTool,
  editXmlBinding,
  treeTool,
  treeXmlBinding,
  grepTool,
  grepXmlBinding,
  viewTool,
  viewXmlBinding,
} from './tools/fs'
import { webSearchTool, webSearchXmlBinding } from './tools/web-search-tool'
import { webFetchTool, webFetchXmlBinding } from './tools/web-fetch-tool'

import { skillTool, skillXmlBinding } from './tools/skill'
import { phaseSubmitTool, phaseSubmitXmlBinding } from './tools/phase-submit'
import { phaseVerdictTool, phaseVerdictXmlBinding } from './tools/phase-verdict'
import {
  clickTool,
  clickXmlBinding,
  doubleClickTool,
  doubleClickXmlBinding,
  rightClickTool,
  rightClickXmlBinding,
  typeTool,
  typeXmlBinding,
  scrollTool,
  scrollXmlBinding,
  dragTool,
  dragXmlBinding,
  navigateTool,
  navigateXmlBinding,
  goBackTool,
  goBackXmlBinding,
  switchTabTool,
  switchTabXmlBinding,
  newTabTool,
  newTabXmlBinding,
  screenshotTool,
  screenshotXmlBinding,
  evaluateTool,
  evaluateXmlBinding,
} from './tools/browser-tools'
import {
  createTaskTool,
  createTaskXmlBinding,
  updateTaskTool,
  updateTaskXmlBinding,
  spawnWorkerTool,
  spawnWorkerXmlBinding,
  killWorkerTool,
  killWorkerXmlBinding,
} from './tools/task-tools'
import {
  agentCreateTool,
  agentCreateXmlBinding,
  agentKillTool,
  agentKillXmlBinding,
} from './tools/agent-tools'

// State models
import { shellModel } from './models/shell'
import { fileReadModel } from './models/file-read'
import { fileWriteModel } from './models/file-write'
import { fileEditModel } from './models/file-edit'
import { fileTreeModel } from './models/file-tree'
import { fileSearchModel } from './models/file-search'
import { fileViewModel } from './models/file-view'
import { webSearchModel } from './models/web-search'
import { webFetchModel } from './models/web-fetch'

import { skillModel } from './models/skill'
import { phaseSubmitModel } from './models/phase-submit'
import { phaseVerdictModel } from './models/phase-verdict'
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

export const catalog = defineCatalog({
  shell: { tool: shellTool, binding: shellXmlBinding, state: shellModel },
  fileRead: { tool: readTool, binding: readXmlBinding, state: fileReadModel },
  fileWrite: { tool: writeTool, binding: writeXmlBinding, state: fileWriteModel },
  fileEdit: { tool: editTool, binding: editXmlBinding, state: fileEditModel },
  fileTree: { tool: treeTool, binding: treeXmlBinding, state: fileTreeModel },
  fileSearch: { tool: grepTool, binding: grepXmlBinding, state: fileSearchModel },
  fileView: { tool: viewTool, binding: viewXmlBinding, state: fileViewModel },
  webSearch: { tool: webSearchTool, binding: webSearchXmlBinding, state: webSearchModel },
  webFetch: { tool: webFetchTool, binding: webFetchXmlBinding, state: webFetchModel },

  skill: { tool: skillTool, binding: skillXmlBinding, state: skillModel },
  phaseSubmit: { tool: phaseSubmitTool, binding: phaseSubmitXmlBinding, state: phaseSubmitModel },
  workflowSubmit: { tool: phaseSubmitTool, binding: phaseSubmitXmlBinding, state: phaseSubmitModel },
  phaseVerdict: { tool: phaseVerdictTool, binding: phaseVerdictXmlBinding, state: phaseVerdictModel },
  click: { tool: clickTool, binding: clickXmlBinding, state: clickModel, group: 'browser' },
  doubleClick: { tool: doubleClickTool, binding: doubleClickXmlBinding, state: doubleClickModel, group: 'browser' },
  rightClick: { tool: rightClickTool, binding: rightClickXmlBinding, state: rightClickModel, group: 'browser' },
  type: { tool: typeTool, binding: typeXmlBinding, state: typeModel, group: 'browser' },
  scroll: { tool: scrollTool, binding: scrollXmlBinding, state: scrollModel, group: 'browser' },
  drag: { tool: dragTool, binding: dragXmlBinding, state: dragModel, group: 'browser' },
  navigate: { tool: navigateTool, binding: navigateXmlBinding, state: navigateModel, group: 'browser' },
  goBack: { tool: goBackTool, binding: goBackXmlBinding, state: goBackModel, group: 'browser' },
  switchTab: { tool: switchTabTool, binding: switchTabXmlBinding, state: switchTabModel, group: 'browser' },
  newTab: { tool: newTabTool, binding: newTabXmlBinding, state: newTabModel, group: 'browser' },
  screenshot: { tool: screenshotTool, binding: screenshotXmlBinding, state: screenshotModel, group: 'browser' },
  evaluate: { tool: evaluateTool, binding: evaluateXmlBinding, state: evaluateModel, group: 'browser' },

  agentCreate: { tool: agentCreateTool, binding: agentCreateXmlBinding, state: agentCreateModel },
  agentKill: { tool: agentKillTool, binding: agentKillXmlBinding, state: agentKillModel },

  createTask: { tool: createTaskTool, binding: createTaskXmlBinding, state: createTaskModel, display: false },
  updateTask: { tool: updateTaskTool, binding: updateTaskXmlBinding, state: updateTaskModel, display: false },
  spawnWorker: { tool: spawnWorkerTool, binding: spawnWorkerXmlBinding, state: spawnWorkerModel, display: false },
  killWorker: { tool: killWorkerTool, binding: killWorkerXmlBinding, state: killWorkerModel, display: false },
} as const)

export type ToolKey = keyof typeof catalog.entries

export function isToolKey(value: string): value is ToolKey {
  return value in catalog.entries
}
