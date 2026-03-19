/**
 * Builder Agent Definition
 *
 * Full read/write agent that implements tasks. Has fs, shell, web search,
 * and parent.message for communicating with the orchestrator. No task management tools.
 */

import { toolSet, defineAgent, continue_, yield_, finish, defineThinkingLens } from '@magnitudedev/agent-definition'
import { readTool, writeTool, editTool, treeTool, searchTool, viewTool } from '../tools/fs'
import { shellBgTool } from '../tools/shell-bg'
import { shellTool } from '../tools/shell'
import { webSearchTool } from '../tools/web-search-tool'
import { webFetchTool } from '../tools/web-fetch-tool'

import { thinkTool } from '../tools/globals'
import { artifactReadTool, artifactWriteTool, artifactUpdateTool } from '../tools/artifact-tools'
import { classifyShellCommand, detectsOutsideCwd, isPathOutsideCwd } from '@magnitudedev/shell-classifier'
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
  artifactRead:   artifactReadTool,
  artifactWrite:  artifactWriteTool,
  artifactUpdate: artifactUpdateTool,

  think:          thinkTool,
})

export const createBuilder = (systemPrompt: string) => defineAgent<typeof tools, PolicyContext>(tools, {
  id: 'builder',
  model: 'secondary',
  systemPrompt,
  thinkingLenses: [qualityLens, turnLens],
  observables: [backgroundProcessesObservable],

  permission: (p) => ({
    shell(input, pctx) {
      const result = classifyShellCommand(input.command)
      if (!pctx.disableShellSafeguards && result.tier === 'forbidden') return p.reject(result.reason ? `This command is forbidden: ${result.reason}` : 'This command is forbidden and cannot be executed.')
      if (!pctx.disableCwdSafeguards && detectsOutsideCwd(input.command, pctx.cwd)) return p.reject('This command targets paths outside the working directory.')
      return p.allow()
    },
    fileWrite(input, ctx) {
      if (!ctx.disableCwdSafeguards && isPathOutsideCwd(input.path, ctx.cwd)) return p.reject('Cannot write to files outside the working directory')
      return p.allow()
    },
    fileEdit(input, ctx) {
      if (!ctx.disableCwdSafeguards && isPathOutsideCwd(input.path, ctx.cwd)) return p.reject('Cannot write to files outside the working directory')
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


  display: (d) => ({
    think() { return d.hidden() },
    _default() { return d.visible() },
  }),
})
