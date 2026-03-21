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
export { browserActionModel, type BrowserActionState } from './browser-action'
export { getModelForToolKey } from './registry'

// Aliases for display compatibility
export { fileWriteModel as contentModel, type FileWriteState as ContentState } from './file-write'
export { fileEditModel as diffModel, type FileEditState as DiffState } from './file-edit'
