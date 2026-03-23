export type { Phase, BaseState, EditDiff } from '@magnitudedev/tools'

export { fileReadModel, type FileReadState } from './file-read'
export { fileWriteModel, type FileWriteState } from './file-write'
export { fileEditModel, type FileEditState } from './file-edit'
export { fileTreeModel, type FileTreeState } from './file-tree'
export { fileSearchModel, type FileSearchState } from './file-search'
export { webFetchModel, type WebFetchState } from './web-fetch'
export { webSearchModel, type WebSearchState } from './web-search'
export { agentCreateModel, type AgentCreateState } from './agent-create'
export { agentKillModel, type AgentKillState } from './agent-kill'
export { skillModel, type SkillState } from './skill'
export { shellModel, type ShellState } from './shell'
export { phaseSubmitModel, type PhaseSubmitState } from './phase-submit'
export { phaseVerdictModel, type PhaseVerdictState } from './phase-verdict'
export {
  createBrowserActionModel,
  type BrowserActionState,
  type BrowserActionModelConfig,
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
} from './browser-action'
export {
  TOOL_DEFINITIONS,
  isToolKey,
  type ToolDefinitionMap,
  type ToolDefinitionFor,
  type ToolKey,
  type ToolModelFor,
  type ToolStateFor,
  type ToolEventFor,
} from '../tools/tool-definitions'

// Aliases for display compatibility
export { fileWriteModel as contentModel, type FileWriteState as ContentState } from './file-write'
export { fileEditModel as diffModel, type FileEditState as DiffState } from './file-edit'
