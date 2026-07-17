/**
 * Per-role toolkit composition.
 *
 * Replaces catalog.ts as the source of tool→state pairings.
 * Each toolkit entry pairs a HarnessTool with its StateModel using the same
 * keys that catalog.ts used (e.g., 'fileRead', 'shell', 'webSearch').
 */

import { defineToolkit, mergeToolkits, type Toolkit, type ToolkitKeys } from '@magnitudedev/harness'
import type { RoleId } from '../agents/role-validation'
import type { ConfigState } from '../ambient/config-ambient'
import { getSlotConfigForRole } from '../ambient/config-ambient'
import type { VcsToolEntry } from '@magnitudedev/vcs'
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

export const taskToolkit = defineToolkit({
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

/** fs + shell + web + task + advisor + goal + skill + compact — base leader toolkit (VCS merged dynamically) */
export const leaderToolkit = mergeToolkits(
  mergeToolkits(workerBase, mergeToolkits(taskToolkit, mergeToolkits(advisorConsultToolkit, goalToolkit))),
  defineToolkit({}),  // placeholder for VCS tools — merged dynamically at runtime
)

// =============================================================================
// Role → Toolkit mapping
// =============================================================================

const ROLE_TOOLKITS: Record<RoleId, Toolkit> = {
  leader:    leaderToolkit,
  engineer:  workerBase,
  artisan:   workerBase,
  scientist: workerBase,
  scout:     workerBase,
  architect: workerBase,
  critic:    criticBase,
  advisor:   compactToolkit,
}

// =============================================================================
// ToolKey — derived from leaderToolkit plus VCS keys
// =============================================================================

/** Tools that should not be displayed in the UI */
export const HIDDEN_TOOLS: ReadonlySet<string> = new Set([
  'createTask', 'updateTask', 'killWorker', 'reassignWorker',
  'messageWorker', 'messageAdvisor', 'finishGoal', 'compact',
])

export type ToolKey = ToolkitKeys<typeof leaderToolkit> | 'checkpointRollback' | 'checkpointChanges'

export function isToolKey(value: string): value is ToolKey {
  return value in leaderToolkit.entries ||
    value === 'checkpointRollback' ||
    value === 'checkpointChanges'
}

// Convert a precise ToolKey to the erased branded type used in events.
// This is a zero-cost cast — the brand is structural and adds no runtime overhead.
export function toToolKeyErased(key: ToolKey): ToolKeyErased {
  return key as ToolKeyErased
}

/**
 * Get the static toolkit for a given role.
 * Returns a Toolkit with entries keyed by the canonical tool keys
 * (fileRead, shell, webSearch, etc.).
 */
function getBaseToolkit(roleId: RoleId): Toolkit {
  return ROLE_TOOLKITS[roleId]
}

/**
 * Merge VCS tool entries into a base toolkit.
 */
function mergeVcsTools(base: Toolkit, vcsEntries: ReadonlyArray<VcsToolEntry>): Toolkit {
  const allEntries: Record<string, any> = { ...base.entries }
  for (const { key, tool } of vcsEntries) {
    allEntries[key] = tool
  }
  return defineToolkit(allEntries)
}

/**
 * Get the effective toolkit for a role, with runtime availability filtering applied.
 *
 * Currently filters:
 * - `fileView`/`queryImage`: one removed based on model vision capability
 * - VCS tools: merged dynamically from the VCS service
 */
export function getEffectiveToolkit(
  roleId: RoleId,
  configState: ConfigState,
  vcsEntries?: ReadonlyArray<VcsToolEntry>,
  options?: { solo?: boolean },
): Toolkit {
  let toolkit = getBaseToolkit(roleId)

  // Merge VCS tools dynamically when provided
  if (roleId === 'leader' && vcsEntries && vcsEntries.length > 0) {
    toolkit = mergeVcsTools(toolkit, vcsEntries)
  }

  // TEMPORARILY DISABLED: advisor consultation.
  // Keep advisorConsultToolkit and static ToolKey support for historical state,
  // but remove messageAdvisor from all runtime tool availability surfaces.
  if (roleId === 'leader' && 'messageAdvisor' in toolkit.entries) {
    toolkit = toolkit.omit('messageAdvisor')
  }

  // Solo mode: remove task/worker tools from the leader toolkit.
  if (roleId === 'leader' && options?.solo) {
    for (const key of ['createTask', 'updateTask', 'spawnWorker', 'killWorker', 'reassignWorker', 'messageWorker'] as const) {
      if (key in toolkit.entries) {
        toolkit = toolkit.omit(key)
      }
    }
  }

  // Vision-based image tool selection
  if ('fileView' in toolkit.entries) {
    const hasVision = getSlotConfigForRole(configState, roleId)?.vision === true
    toolkit = hasVision
      ? toolkit.omit('queryImage')
      : toolkit.omit('fileView')
  }

  return toolkit
}
