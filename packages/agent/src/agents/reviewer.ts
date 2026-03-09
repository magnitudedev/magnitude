/**
 * Reviewer Agent Definition
 *
 * Independently verifies implemented changes meet the user's intent.
 * Read-only codebase access + shell for running tests/builds.
 * No file write access — reviewers only observe, test, and report.
 */

import { toolSet, defineAgent, continue_, yield_, finish, taskThinkingLens, turnThinkingLens } from '@magnitudedev/agent-definition'
import { readTool, treeTool, searchTool } from '../tools/fs'
import { shellTool } from '../tools/shell'

import { thinkTool } from '../tools/globals'
import { artifactReadTool } from '../tools/artifact-tools'
import { agentCreateTool, agentDismissTool } from '../tools/agent-tools'
import { classifyShellCommand, detectsOutsideCwd } from '@magnitude/shell-classifier'
import type { PolicyContext } from './types'

const tools = toolSet({
  fileRead:       readTool,
  fileTree:       treeTool,
  fileSearch:     searchTool,
  shell:          shellTool,
  artifactRead:   artifactReadTool,

  agentCreate:    agentCreateTool,
  agentDismiss:   agentDismissTool,

  think:          thinkTool,
})

export const createReviewer = (systemPrompt: string) => defineAgent<typeof tools, PolicyContext>(tools, {
  id: 'reviewer',
  model: 'secondary',
  systemPrompt,
  thinkingLenses: [taskThinkingLens, turnThinkingLens],

  permission: (p) => ({
    shell(input, ctx) {
      const result = classifyShellCommand(input.command)
      if (result.tier === 'forbidden') return p.reject(result.reason ? `This command is forbidden: ${result.reason}` : 'This command is forbidden.')
      if (detectsOutsideCwd(input.command, ctx.cwd)) return p.reject('This command targets paths outside the working directory.')
      return p.allow()
    },
    _default() { return p.allow() },
  }),

  turn: {
    decide(turnCtx) {
      if (turnCtx.cancelled) return finish()
      if (turnCtx.error) return continue_()
      if (turnCtx.toolsCalled.length === 0 && turnCtx.messagesSent.some(m => m.dest === 'parent')) return yield_()
      if (turnCtx.toolsCalled.some(t => t === 'agentCreate')) return yield_()
      return continue_()
    },
  },


  display: (d) => ({
    think() { return d.hidden() },
    _default() { return d.visible() },
  }),
})
