/**
 * Planner Agent Definition
 *
 * Read-only agent that produces implementation plans and makes decisions.
 * Uses primary model (planning needs reasoning power).
 * Communicates back via parent.message.
 */

import { toolSet, defineAgent, continue_, yield_, finish, defineThinkingLens } from '@magnitudedev/agent-definition'
import { readTool, treeTool, searchTool } from '../tools/fs'
import { shellTool } from '../tools/shell'
import { webSearchTool } from '../tools/web-search-tool'
import { webFetchTool } from '../tools/web-fetch-tool'

import { thinkTool } from '../tools/globals'
// import { gatherTool } from '../tools/gather'
import { artifactReadTool, artifactWriteTool, artifactUpdateTool } from '../tools/artifact-tools'
import { classifyShellCommand } from '@magnitudedev/shell-classifier'
import type { PolicyContext } from './types'
import { backgroundProcessesObservable } from '../observables/background-processes-observable'

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

const tools = toolSet({
  fileRead:     readTool,
  fileTree:     treeTool,
  fileSearch:   searchTool,
  shell:        shellTool,
  webSearch:    webSearchTool,
  webFetch:     webFetchTool,
  // gather:       gatherTool,
  artifactRead: artifactReadTool,
  artifactWrite: artifactWriteTool,
  artifactUpdate: artifactUpdateTool,

  think:         thinkTool,
})

export const createPlanner = (systemPrompt: string) => defineAgent<typeof tools, PolicyContext>(tools, {
  id: 'planner',
  model: 'secondary',
  systemPrompt,
  thinkingLenses: [ideateLens, velocityLens, alignmentLens, capacityLens, turnLens],
  observables: [backgroundProcessesObservable],

  permission: (p) => ({
    shell(input) {
      const result = classifyShellCommand(input.command)
      if (result.tier === 'readonly') return p.allow()
      // Planners are read-only — reject anything above readonly
      return p.reject('Planners can only run read-only shell commands (ls, cat, git log, etc).')
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
