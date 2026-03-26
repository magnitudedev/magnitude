import {
  shellModel, fileReadModel, fileWriteModel, fileEditModel,
  fileTreeModel, fileSearchModel, webSearchModel, webFetchModel,
  agentCreateModel, agentKillModel, skillModel,
  clickModel, doubleClickModel, rightClickModel, typeModel, scrollModel, dragModel,
  navigateModel, goBackModel, switchTabModel, newTabModel, screenshotModel, evaluateModel,
  phaseSubmitModel, phaseVerdictModel, shellBgModel, fileViewModel,
} from '../models'

export const TOOL_DEFINITIONS = {
  shell: { key: 'shell', model: shellModel },
  fileRead: { key: 'fileRead', model: fileReadModel },
  fileWrite: { key: 'fileWrite', model: fileWriteModel },
  fileEdit: { key: 'fileEdit', model: fileEditModel },
  fileTree: { key: 'fileTree', model: fileTreeModel },
  fileSearch: { key: 'fileSearch', model: fileSearchModel },
  webSearch: { key: 'webSearch', model: webSearchModel },
  webFetch: { key: 'webFetch', model: webFetchModel },
  agentCreate: { key: 'agentCreate', model: agentCreateModel, display: false },
  agentKill: { key: 'agentKill', model: agentKillModel, display: false },
  skill: { key: 'skill', model: skillModel },
  shellBg: { key: 'shellBg', model: shellBgModel },
  fileView: { key: 'fileView', model: fileViewModel },
  click: { key: 'click', model: clickModel, group: 'browser' },
  doubleClick: { key: 'doubleClick', model: doubleClickModel, group: 'browser' },
  rightClick: { key: 'rightClick', model: rightClickModel, group: 'browser' },
  type: { key: 'type', model: typeModel, group: 'browser' },
  scroll: { key: 'scroll', model: scrollModel, group: 'browser' },
  drag: { key: 'drag', model: dragModel, group: 'browser' },
  navigate: { key: 'navigate', model: navigateModel, group: 'browser' },
  goBack: { key: 'goBack', model: goBackModel, group: 'browser' },
  switchTab: { key: 'switchTab', model: switchTabModel, group: 'browser' },
  newTab: { key: 'newTab', model: newTabModel, group: 'browser' },
  screenshot: { key: 'screenshot', model: screenshotModel, group: 'browser' },
  evaluate: { key: 'evaluate', model: evaluateModel, group: 'browser' },
  phaseSubmit: { key: 'phaseSubmit', model: phaseSubmitModel },
  workflowSubmit: { key: 'workflowSubmit', model: phaseSubmitModel },
  phaseVerdict: { key: 'phaseVerdict', model: phaseVerdictModel },
} as const

export type ToolDefinitionMap = typeof TOOL_DEFINITIONS
export type ToolKey = keyof ToolDefinitionMap
export type ToolDefinitionFor<K extends ToolKey> = ToolDefinitionMap[K]
export type ToolModelFor<K extends ToolKey> = ToolDefinitionFor<K>['model']
export type ToolStateFor<K extends ToolKey> = ToolDefinitionFor<K>['model']['initial']
export type ToolState = ToolStateFor<ToolKey>
export type BrowserToolKey = {
  [K in ToolKey]: ToolDefinitionMap[K] extends { group: 'browser' } ? K : never
}[ToolKey]
export type ToolEventFor<K extends ToolKey> = Parameters<ToolDefinitionFor<K>['model']['reduce']>[1]
export function isToolKey(value: string): value is ToolKey {
  return Object.prototype.hasOwnProperty.call(TOOL_DEFINITIONS, value)
}
