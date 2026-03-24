/**
 * Builder Agent Definition
 *
 * Full read/write agent that implements tasks. Has fs, shell, web search,
 * and parent.message for communicating with the orchestrator. No task management tools.
 */

import { toolSet, defineRole, continue_, yield_, finish, defineThinkingLens } from '@magnitudedev/roles'
import builderPromptRaw from './prompts/builder.txt' with { type: 'text' }
import { compilePromptTemplate } from '../prompts/system-prompt'
import { readTool, writeTool, editTool, treeTool, searchTool, viewTool } from '../tools/fs'
import { shellBgTool } from '../tools/shell-bg'
import { shellTool } from '../tools/shell'
import { webSearchTool } from '../tools/web-search-tool'
import { webFetchTool } from '../tools/web-fetch-tool'

import { phaseSubmitTool } from '../tools/globals'
import { classifyShellCommand, writesStayWithin, isPathWithin } from '@magnitudedev/shell-classifier'
import type { PolicyContext } from './types'
import { backgroundProcessesObservable } from '../observables/background-processes-observable'

const qualityLens = defineThinkingLens({
  name: 'quality',
  trigger: 'When writing or modifying code',
  description: "Consider code quality and adherence to existing patterns. Does this match the conventions, abstractions, and style already in use? Is this consistent with the surrounding codebase? Don't just make it work — make it fit.",
})

const turnLens = defineThinkingLens({
  name: 'turn',
  trigger: 'When planning your next actions',
  description: 'Plan what to read and edit this turn. What files do you need to understand before making changes? What\'s the right order of edits?',
})

const systemPrompt = compilePromptTemplate(builderPromptRaw)

const tools = toolSet({
  fileRead:       readTool,
  fileWrite:      writeTool,
  fileEdit:       editTool,
  fileTree:       treeTool,
  fileSearch:     searchTool,
  fileView:       viewTool,
  shell:          shellTool,
  shellBg:        shellBgTool,
  webSearch:      webSearchTool,
  webFetch:       webFetchTool,

  phaseSubmit:    phaseSubmitTool,
})

export const builderRole = defineRole<typeof tools, 'builder', PolicyContext>({
  tools,
  id: 'builder',
  slot: 'builder',
  systemPrompt,
  lenses: [qualityLens, turnLens],
  defaultRecipient: 'parent',
  protocolRole: 'subagent',
  initialContext: { parentConversation: true },
  spawnable: true,
  observables: [backgroundProcessesObservable],

  permission: (p) => ({
    shell(input, pctx) {
      const result = classifyShellCommand(input.command)
      const allowedPrefixes = [pctx.workspacePath]
      if (!pctx.disableShellSafeguards && result.tier === 'forbidden') return p.reject(result.reason ? `This command is forbidden: ${result.reason}` : 'This command is forbidden and cannot be executed.')
      if (!pctx.disableCwdSafeguards && !writesStayWithin(input.command, pctx.cwd, ...(allowedPrefixes ?? []))) return p.reject('This command targets paths outside the working directory.')
      return p.allow()
    },
    fileWrite(input, ctx) {
      const allowedPrefixes = [ctx.workspacePath]
      if (!ctx.disableCwdSafeguards && !isPathWithin(input.path, ctx.cwd, ...(allowedPrefixes ?? []))) return p.reject('Cannot write to files outside the working directory')
      return p.allow()
    },
    fileEdit(input, ctx) {
      const allowedPrefixes = [ctx.workspacePath]
      if (!ctx.disableCwdSafeguards && !isPathWithin(input.path, ctx.cwd, ...(allowedPrefixes ?? []))) return p.reject('Cannot write to files outside the working directory')
      return p.allow()
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
})
