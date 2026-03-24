/**
 * Planner Agent Definition
 *
 * Read-only agent that produces implementation plans and makes decisions.
 * Uses primary model (planning needs reasoning power).
 * Communicates back via parent.message.
 */

import { resolve } from 'node:path'
import { toolSet, defineRole, continue_, yield_, finish, defineThinkingLens } from '@magnitudedev/roles'
import plannerPromptRaw from './prompts/planner.txt' with { type: 'text' }
import { compilePromptTemplate } from '../prompts/system-prompt'
import { readTool, writeTool, editTool, treeTool, searchTool } from '../tools/fs'
import { shellBgTool } from '../tools/shell-bg'
import { shellTool } from '../tools/shell'
import { webSearchTool } from '../tools/web-search-tool'
import { webFetchTool } from '../tools/web-fetch-tool'


// import { gatherTool } from '../tools/gather'
import { classifyShellCommand, writesStayWithin, isPathWithin } from '@magnitudedev/shell-classifier'
import type { PolicyContext } from './types'
import { backgroundProcessesObservable } from '../observables/background-processes-observable'
import { expandWorkspacePath } from '../workspace/workspace-path'

const ideateLens = defineThinkingLens({
  name: 'ideate',
  trigger: 'When considering how to approach the implementation',
  description: "Brainstorm approaches freely. What are the different ways this could be implemented? Don't evaluate yet — generate options and explore the solution space.",
})

const velocityLens = defineThinkingLens({
  name: 'velocity',
  trigger: 'When evaluating an implementation approach',
  description: "What's the fastest way to make this work? Minimize files touched, minimize changes to existing code. Consider the simplest path to a working solution.",
})

const alignmentLens = defineThinkingLens({
  name: 'alignment',
  trigger: 'When evaluating an implementation approach',
  description: 'What approach best harmonizes with the existing system? What patterns, abstractions, and mechanisms already exist that this change should participate in?',
})

const capacityLens = defineThinkingLens({
  name: 'capacity',
  trigger: 'When evaluating an implementation approach',
  description: 'What approach leaves the system best able to handle this kind of change going forward? Imagine five more similar changes — what would make all of them natural?',
})

const turnLens = defineThinkingLens({
  name: 'turn',
  trigger: 'When planning your next actions',
  description: 'Plan what to read, explore, or write this turn. What information do you still need? Are you ready to write the plan, or do you need to investigate more?',
})

const systemPrompt = compilePromptTemplate(plannerPromptRaw)

const tools = toolSet({
  fileRead:     readTool,
  fileWrite:    writeTool,
  fileEdit:     editTool,
  fileTree:     treeTool,
  fileSearch:   searchTool,
  shell:        shellTool,
  shellBg:      shellBgTool,
  webSearch:    webSearchTool,
  webFetch:     webFetchTool,
  // gather:       gatherTool,

})

export const plannerRole = defineRole<typeof tools, 'planner', PolicyContext>({
  tools,
  id: 'planner',
  slot: 'planner',
  systemPrompt,
  lenses: [ideateLens, velocityLens, alignmentLens, capacityLens, turnLens],
  defaultRecipient: 'parent',
  protocolRole: 'subagent',
  initialContext: { parentConversation: true },
  spawnable: true,
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
})
