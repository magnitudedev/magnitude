/**
 * Per-role toolkit composition.
 *
 * Replaces catalog.ts as the source of tool→state pairings.
 * Each toolkit entry pairs a HarnessTool with its StateModel using the same
 * keys that catalog.ts used (e.g., 'fileRead', 'shell', 'webSearch').
 */

import { defineToolkit, mergeToolkits, type Toolkit, type ToolkitEntry, type ToolkitKeys } from '@magnitudedev/harness'
import type { RoleId } from '../agents/role-validation'
import type { ConfigState } from '../ambient/config-ambient'
import { ROLE_TO_SLOT } from '@magnitudedev/roles'
import { vcsToolkit } from '@magnitudedev/vcs'
import type { ToolKeyErased } from './types'

// --- Tools ---
import { readTool, writeTool, editTool, treeTool, grepTool, viewTool } from './fs'
import { queryImageTool } from './query-image'
import { shellTool } from './shell'
import { webSearchTool } from './web-search'
import { webFetchTool } from './web-fetch-tool'
import { createTaskTool, updateTaskTool, spawnWorkerTool, killWorkerTool, reassignWorkerTool } from './task-tools'
import { skillTool } from './skill-tool'
import { messageWorkerTool } from './agent-communication'
import { messageAdvisorTool } from './advisor'
import { finishGoalTool } from './goal'

// --- State Models ---
import { fileReadModel } from '../models/file-read'
import { fileWriteModel } from '../models/file-write'
import { fileEditModel } from '../models/file-edit'
import { fileTreeModel } from '../models/file-tree'
import { fileSearchModel } from '../models/file-search'
import { fileViewModel } from '../models/file-view'
import { queryImageModel } from '../models/query-image'
import { shellModel } from '../models/shell'
import { webSearchModel } from '../models/web-search'
import { webFetchModel } from '../models/web-fetch'
import { createTaskModel } from '../models/create-task'
import { updateTaskModel } from '../models/update-task'
import { spawnWorkerModel } from '../models/spawn-worker'
import { killWorkerModel } from '../models/kill-worker'
import { reassignWorkerModel } from '../models/reassign-worker'
import { skillActivationModel } from '../models/skill-activation'
import { messageWorkerModel } from '../models/message-worker'
import { messageAdvisorModel } from '../models/message-advisor'
import { compactTool } from './compact'
import { compactModel } from '../models/compact'
import { finishGoalModel } from '../models/finish-goal'

// =============================================================================
// Group Toolkits
// =============================================================================

export const fsToolkit = defineToolkit({
  fileRead:   { tool: readTool,   state: fileReadModel },
  fileWrite:  { tool: writeTool,  state: fileWriteModel },
  fileEdit:   { tool: editTool,   state: fileEditModel },
  fileTree:   { tool: treeTool,   state: fileTreeModel },
  fileSearch: { tool: grepTool,   state: fileSearchModel },
  fileView:   { tool: viewTool,   state: fileViewModel },
  queryImage: { tool: queryImageTool, state: queryImageModel },
})

export const shellToolkit = defineToolkit({
  shell: { tool: shellTool, state: shellModel },
})

export const webToolkit = defineToolkit({
  webSearch: { tool: webSearchTool, state: webSearchModel },
  webFetch:  { tool: webFetchTool,  state: webFetchModel },
})

type TaskToolkitEntries = {
  readonly createTask: ToolkitEntry
  readonly updateTask: ToolkitEntry
  readonly spawnWorker: ToolkitEntry
  readonly killWorker: ToolkitEntry
  readonly reassignWorker: ToolkitEntry
  readonly messageWorker: ToolkitEntry
}

export const taskToolkit: Toolkit<TaskToolkitEntries> = defineToolkit({
  createTask:      { tool: createTaskTool,      state: createTaskModel },
  updateTask:      { tool: updateTaskTool,      state: updateTaskModel },
  spawnWorker:     { tool: spawnWorkerTool,     state: spawnWorkerModel },
  killWorker:      { tool: killWorkerTool,      state: killWorkerModel },
  reassignWorker:  { tool: reassignWorkerTool,  state: reassignWorkerModel },
  messageWorker:   { tool: messageWorkerTool,  state: messageWorkerModel },
})

export const skillToolkit = defineToolkit({
  skill: { tool: skillTool, state: skillActivationModel },
})

export const advisorConsultToolkit = defineToolkit({
  messageAdvisor: { tool: messageAdvisorTool, state: messageAdvisorModel },
})

export const goalToolkit = defineToolkit({
  finishGoal: { tool: finishGoalTool, state: finishGoalModel },
})

export const compactToolkit = defineToolkit({
  compact: { tool: compactTool, state: compactModel },
})

// =============================================================================
// Composite Toolkits
// =============================================================================

/** fs + shell + web + skill + compact — shared by most worker roles */
const workerBase = mergeToolkits(
  mergeToolkits(fsToolkit, shellToolkit),
  mergeToolkits(webToolkit, mergeToolkits(skillToolkit, compactToolkit)),
)

/** fs + shell + skill + compact — no web access */
const criticBase = mergeToolkits(
  mergeToolkits(fsToolkit, shellToolkit),
  mergeToolkits(skillToolkit, compactToolkit),
)

/** fs + shell + web + task + advisor + goal + skill + compact. */
export const leaderToolkit = mergeToolkits(
  workerBase,
  mergeToolkits(taskToolkit, mergeToolkits(advisorConsultToolkit, goalToolkit)),
)

/** Complete executable tool universe understood by this agent runtime. */
export const toolUniverseToolkit = mergeToolkits(leaderToolkit, vcsToolkit)

// =============================================================================
// Role → Toolkit mapping
// =============================================================================

// =============================================================================
// ToolKey — derived from the complete executable universe
// =============================================================================

/** Tools that should not be displayed in the UI */
export const HIDDEN_TOOLS: ReadonlySet<string> = new Set([
  'createTask', 'updateTask', 'killWorker', 'reassignWorker',
  'messageWorker', 'messageAdvisor', 'finishGoal', 'compact',
])

export type ToolKey = ToolkitKeys<typeof toolUniverseToolkit>

export function isToolKey(value: string): value is ToolKey {
  return value in toolUniverseToolkit.entries
}

export interface AgentToolSelectionInput {
  readonly roleId: RoleId
  readonly configState: ConfigState
  readonly solo: boolean
  readonly vcsAvailable: boolean
}

function baseToolKeys(roleId: RoleId): readonly ToolKey[] {
  switch (roleId) {
    case 'leader':
      return leaderToolkit.keys
    case 'critic':
      return criticBase.keys
    case 'advisor':
      return compactToolkit.keys
    default:
      return workerBase.keys
  }
}

/** Sole policy for selecting ordered tool keys for an agent fork. */
export function selectAgentToolKeys(input: AgentToolSelectionInput): readonly ToolKey[] {
  const { roleId, configState, solo, vcsAvailable } = input
  let keys = [...baseToolKeys(roleId)]

  if (roleId === 'leader') {
    keys = keys.filter(key => key !== 'messageAdvisor')
    if (solo) {
      const soloExcluded = new Set<ToolKey>(['createTask', 'updateTask', 'spawnWorker', 'killWorker', 'reassignWorker', 'messageWorker'])
      keys = keys.filter(key => !soloExcluded.has(key))
    }
    if (vcsAvailable) keys.push('checkpointRollback', 'checkpointChanges')
  }

  const activeSlotId = ROLE_TO_SLOT[roleId]
  const activeSlot = configState.bySlot[activeSlotId]
  const activeHasVision = activeSlot._tag === 'Ready' && activeSlot.config.vision === true
  // Opposite-slot vision fallback is temporarily disabled with the secondary model.
  // const otherSlotId = activeSlotId === 'primary' ? 'secondary' : 'primary'
  // const otherSlot = configState.bySlot[otherSlotId]
  // const otherHasVision = otherSlot._tag === 'Ready' && otherSlot.config.vision === true
  keys = keys.filter(key => key !== 'fileView' && key !== 'queryImage')
  if (activeHasVision) keys.push('fileView')
  // else if (otherHasVision) keys.push('queryImage')

  return keys
}

export function materializeAgentToolkit(universe: Toolkit, toolKeys: readonly string[]): Toolkit {
  const missing = toolKeys.filter(key => !(key in universe.entries))
  if (missing.length > 0) throw new Error(`Tool universe is missing selected keys: ${missing.join(', ')}`)
  let bySelection = materializedToolkits.get(universe)
  if (!bySelection) {
    bySelection = new Map()
    materializedToolkits.set(universe, bySelection)
  }
  const selection = JSON.stringify(toolKeys)
  const existing = bySelection.get(selection)
  if (existing) return existing
  const toolkit = universe.pick(...toolKeys)
  bySelection.set(selection, toolkit)
  return toolkit
}

const materializedToolkits = new WeakMap<Toolkit, Map<string, Toolkit>>()

// Convert a precise ToolKey to the erased branded type used in events.
// This is a zero-cost cast — the brand is structural and adds no runtime overhead.
export function toToolKeyErased(key: ToolKey): ToolKeyErased {
  return key as ToolKeyErased
}
