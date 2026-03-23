import {
  shellModel, fileReadModel, fileWriteModel, fileEditModel,
  fileTreeModel, fileSearchModel, webSearchModel, webFetchModel,
  agentCreateModel, agentKillModel, skillModel,
  clickModel, doubleClickModel, rightClickModel, typeModel, scrollModel, dragModel,
  navigateModel, goBackModel, switchTabModel, newTabModel, screenshotModel, evaluateModel,
  phaseSubmitModel, phaseVerdictModel,
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
  click: { key: 'click', model: clickModel },
  doubleClick: { key: 'doubleClick', model: doubleClickModel },
  rightClick: { key: 'rightClick', model: rightClickModel },
  type: { key: 'type', model: typeModel },
  scroll: { key: 'scroll', model: scrollModel },
  drag: { key: 'drag', model: dragModel },
  navigate: { key: 'navigate', model: navigateModel },
  goBack: { key: 'goBack', model: goBackModel },
  switchTab: { key: 'switchTab', model: switchTabModel },
  newTab: { key: 'newTab', model: newTabModel },
  screenshot: { key: 'screenshot', model: screenshotModel },
  evaluate: { key: 'evaluate', model: evaluateModel },
  'phase-submit': { key: 'phase-submit', model: phaseSubmitModel },
  'workflow-submit': { key: 'workflow-submit', model: phaseSubmitModel },
  'phase-verdict': { key: 'phase-verdict', model: phaseVerdictModel },
} as const

export type ToolDefinitionMap = typeof TOOL_DEFINITIONS
export type ToolKey = keyof ToolDefinitionMap
export type ToolDefinitionFor<K extends ToolKey> = ToolDefinitionMap[K]
export type ToolModelFor<K extends ToolKey> = ToolDefinitionFor<K>['model']
export type ToolStateFor<K extends ToolKey> = ToolDefinitionFor<K>['model']['initial']
export type ToolEventFor<K extends ToolKey> = Parameters<ToolDefinitionFor<K>['model']['reduce']>[1]
export function isToolKey(value: string): value is ToolKey {
  return Object.prototype.hasOwnProperty.call(TOOL_DEFINITIONS, value)
}