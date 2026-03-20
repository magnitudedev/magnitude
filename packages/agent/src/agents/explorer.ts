/**
 * Explorer Agent Definition
 *
 * Read-only agent that answers informational questions by exploring the codebase.
 * Uses secondary model. Communicates back via parent.message.
 */

import { resolve } from 'node:path'
import { toolSet, defineAgent, continue_, yield_, finish, defineThinkingLens } from '@magnitudedev/agent-definition'
import { readTool, writeTool, editTool, treeTool, searchTool, viewTool } from '../tools/fs'
import { shellBgTool } from '../tools/shell-bg'
import { shellTool } from '../tools/shell'
import { webSearchTool } from '../tools/web-search-tool'
import { webFetchTool } from '../tools/web-fetch-tool'

import { thinkTool } from '../tools/globals'
import { classifyShellCommand, writesStayWithin, isPathWithin } from '@magnitudedev/shell-classifier'
import type { PolicyContext } from './types'
import { backgroundProcessesObservable } from '../observables/background-processes-observable'
import { expandWorkspacePath } from '../workspace/workspace-path'

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

  think:         thinkTool,
})

export const createExplorer = (systemPrompt: string) => defineAgent<typeof tools, PolicyContext>(tools, {
  id: 'explorer',
  model: 'secondary',
  systemPrompt,
  thinkingLenses: [strategyLens, turnLens],
  observables: [backgroundProcessesObservable],

  permission: (p) => ({
    shell(input, ctx) {
      const result = classifyShellCommand(input.command)
      if (result.tier === 'readonly') return p.allow()
      if (result.tier === 'forbidden') return p.reject(result.reason ? `This command is forbidden: ${result.reason}` : 'This command is forbidden.')
      if (writesStayWithin(input.command, ctx.workspacePath)) return p.allow()
      return p.reject('Explorers can only run read-only shell commands, or write to the workspace ($M/).')
    },
    fileWrite(input, ctx) {
      const expanded = expandWorkspacePath(input.path, ctx.workspacePath)
      const resolved = resolve(ctx.cwd, expanded)
      if (isPathWithin(resolved, ctx.workspacePath)) return p.allow()
      return p.reject('Explorers can only write to the workspace ($M/).')
    },
    fileEdit(input, ctx) {
      const expanded = expandWorkspacePath(input.path, ctx.workspacePath)
      const resolved = resolve(ctx.cwd, expanded)
      if (isPathWithin(resolved, ctx.workspacePath)) return p.allow()
      return p.reject('Explorers can only edit files in the workspace ($M/).')
    },
    _default() { return p.allow() },
  }),

  turn: {
    decide(turnCtx) {
      if (turnCtx.cancelled) return finish()
      if (turnCtx.error) return continue_()
      if (turnCtx.toolsCalled.length === 0 && turnCtx.messagesSent.some(m => m.dest === 'parent')) return yield_()
      return continue_()
    },
  },


  display: (d) => ({
    think() { return d.hidden() },
    _default() { return d.visible() },
  }),
})