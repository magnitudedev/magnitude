/**
 * Scout Agent Definition
 *
 * Read-only agent that evaluates implementation approaches by exploring the codebase
 * and analyzing trade-offs along three dimensions: velocity, alignment, capacity.
 * Uses secondary model. Communicates back via parent.message.
 */

import { toolSet, defineAgent, continue_, yield_, finish, taskThinkingLens, turnThinkingLens } from '@magnitudedev/agent-definition'
import { readTool, treeTool, searchTool } from '../tools/fs'
import { shellTool } from '../tools/shell'

import { thinkTool } from '../tools/globals'
import { artifactReadTool, artifactWriteTool } from '../tools/artifact-tools'
import { classifyShellCommand } from '@magnitude/shell-classifier'
import type { PolicyContext } from './types'

const tools = toolSet({
  fileRead:      readTool,
  fileTree:      treeTool,
  fileSearch:    searchTool,
  shell:         shellTool,
  artifactRead:  artifactReadTool,
  artifactWrite: artifactWriteTool,

  think:         thinkTool,
})

export const createScout = (systemPrompt: string) => defineAgent<typeof tools, PolicyContext>(tools, {
  id: 'scout',
  model: 'secondary',
  systemPrompt,
  thinkingLenses: [taskThinkingLens, turnThinkingLens],

  permission: (p) => ({
    shell(input) {
      const result = classifyShellCommand(input.command)
      if (result.tier === 'readonly') return p.allow()
      // Scouts are read-only — reject anything above readonly
      return p.reject('Scouts can only run read-only shell commands (ls, cat, git log, etc).')
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
