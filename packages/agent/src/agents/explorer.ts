/**
 * Explorer Agent Definition
 *
 * Agent that answers informational questions by exploring the codebase.
 * Has read-only shell access, but can write files within the workspace.
 * Uses secondary model. Communicates back via parent.message.
 */

import { toolSet, defineRole, continue_, yield_, finish, defineThinkingLens } from '@magnitudedev/roles'
import explorerPromptRaw from './prompts/explorer.txt' with { type: 'text' }
import { compilePromptTemplate } from '../prompts/system-prompt'
import { readTool, writeTool, editTool, treeTool, searchTool, viewTool } from '../tools/fs'
import { shellBgTool } from '../tools/shell-bg'
import { shellTool } from '../tools/shell'
import { webSearchTool } from '../tools/web-search-tool'
import { webFetchTool } from '../tools/web-fetch-tool'

import { allowReadonlyShell, denyForbiddenCommands, denyMutatingGit, denyWritesOutside, allowAll } from './policy'
import type { PolicyContext } from './types'
import { backgroundProcessesObservable } from '../observables/background-processes-observable'

const strategyLens = defineThinkingLens({
  name: 'strategy',
  trigger: 'When deciding what to investigate next',
  description: "How can you gather the needed information quickly and efficiently? What tools and techniques will get you there fastest — tree, search, read, shell, web? Prioritize high-signal sources. Don't read aimlessly.",
})

const turnLens = defineThinkingLens({
  name: 'turn',
  trigger: 'When planning your next actions',
  description: 'Plan what to read, search, or explore this turn. Maximize coverage per turn by reading multiple files in parallel.',
})

const systemPrompt = compilePromptTemplate(explorerPromptRaw)

const tools = toolSet({
  fileRead:      readTool,
  fileWrite:     writeTool,
  fileEdit:      editTool,
  fileTree:      treeTool,
  fileSearch:    searchTool,
  fileView:      viewTool,
  shell:         shellTool,
  shellBg:       shellBgTool,
  webSearch:     webSearchTool,
  webFetch:      webFetchTool,

})

export const explorerRole = defineRole<typeof tools, 'explorer', PolicyContext>({
  tools,
  id: 'explorer',
  slot: 'explorer',
  systemPrompt,
  lenses: [strategyLens, turnLens],
  defaultRecipient: 'parent',
  protocolRole: 'subagent',
  initialContext: { parentConversation: true },
  spawnable: true,
  observables: [backgroundProcessesObservable],
  lifecyclePrompts: {
    parentOnSpawn: 'If you need context on multiple areas, spawn additional explorers in parallel rather than waiting for one at a time.',
    parentOnIdle: 'Evaluate whether the explorer\'s findings are sufficient. If ambiguities or unknowns remain, send the explorer back with specific questions or spawn additional explorers. Do not proceed to planning or building with incomplete context.',
  },

  policy: [
    allowReadonlyShell(),
    denyForbiddenCommands(),
    denyMutatingGit(),
    denyWritesOutside(ctx => [ctx.workspacePath]),
    allowAll(),
  ],

  turn: {
    decide(turnCtx) {
      if (turnCtx.cancelled) return finish()
      if (turnCtx.error) return continue_()
      if (turnCtx.toolsCalled.length === 0 && turnCtx.messagesSent.some(m => m.dest === 'parent')) return yield_()
      return continue_()
    },
  },
})