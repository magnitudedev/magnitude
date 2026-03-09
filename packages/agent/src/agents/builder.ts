/**
 * Builder Agent Definition
 *
 * Full read/write agent that implements tasks. Has fs, shell, web search,
 * and parent.message for communicating with the orchestrator. No task management tools.
 */

import { toolSet, defineAgent, continue_, yield_, finish, taskThinkingLens, turnThinkingLens } from '@magnitudedev/agent-definition'
import { readTool, writeTool, editTool, treeTool, searchTool } from '../tools/fs'
import { shellTool } from '../tools/shell'
import { webSearchTool } from '../tools/web-search-tool'
import { webFetchTool } from '../tools/web-fetch-tool'

import { thinkTool } from '../tools/globals'
import { artifactReadTool, artifactWriteTool, artifactUpdateTool } from '../tools/artifact-tools'
import { classifyShellCommand, detectsOutsideCwd, isPathOutsideCwd } from '@magnitude/shell-classifier'
import type { PolicyContext } from './types'

const tools = toolSet({
  fileRead:       readTool,
  fileWrite:      writeTool,
  fileEdit:       editTool,
  fileTree:       treeTool,
  fileSearch:     searchTool,
  shell:          shellTool,
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
  thinkingLenses: [taskThinkingLens, turnThinkingLens],

  permission: (p) => ({
    shell(input, pctx) {
      const result = classifyShellCommand(input.command)
      if (result.tier === 'forbidden') return p.reject(result.reason ? `This command is forbidden: ${result.reason}` : 'This command is forbidden and cannot be executed.')
      if (detectsOutsideCwd(input.command, pctx.cwd)) return p.reject('This command targets paths outside the working directory.')
      return p.allow()
    },
    fileWrite(input, ctx) {
      if (isPathOutsideCwd(input.path, ctx.cwd)) return p.reject('Cannot write to files outside the working directory')
      return p.allow()
    },
    fileEdit(input, ctx) {
      if (isPathOutsideCwd(input.path, ctx.cwd)) return p.reject('Cannot write to files outside the working directory')
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
