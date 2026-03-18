/**
 * Debugger Agent Definition
 *
 * Root-cause investigation agent with full read/write + shell access.
 * Focuses on diagnosis — forming hypotheses, testing them, narrowing down causes.
 */

import { toolSet, defineAgent, continue_, yield_, finish, defineThinkingLens } from '@magnitudedev/agent-definition'
import { readTool, writeTool, editTool, treeTool, searchTool } from '../tools/fs'
import { shellTool } from '../tools/shell'
import { webSearchTool } from '../tools/web-search-tool'
import { webFetchTool } from '../tools/web-fetch-tool'

import { thinkTool } from '../tools/globals'
import { artifactReadTool, artifactWriteTool } from '../tools/artifact-tools'
import { classifyShellCommand, detectsOutsideCwd, isPathOutsideCwd } from '@magnitudedev/shell-classifier'
import type { PolicyContext } from './types'
import { backgroundProcessesObservable } from '../observables/background-processes-observable'

const hypothesisLens = defineThinkingLens({
  name: 'hypothesis',
  trigger: 'When investigating a bug or unexpected behavior',
  description: "State your current hypothesis for the root cause. What evidence supports it? What evidence contradicts it? If you don't have a hypothesis yet, form one from the symptoms.",
})

const skepticismLens = defineThinkingLens({
  name: 'skepticism',
  trigger: 'After forming or updating a hypothesis',
  description: 'Question your hypothesis. Is that really the root cause, or just where the symptom manifests? Could something else explain the evidence? What would disprove your current theory?',
})

const strategyLens = defineThinkingLens({
  name: 'strategy',
  trigger: 'When deciding how to investigate further',
  description: "How can you best collect evidence to prove or disprove your hypothesis? What commands, logs, or code paths should you examine? Design targeted experiments — don't just read code and guess.",
})

const turnLens = defineThinkingLens({
  name: 'turn',
  trigger: 'When planning your next actions',
  description: "Plan what to run, read, or modify this turn. What's the most informative next step to test your hypothesis?",
})

const tools = toolSet({
  fileRead:      readTool,
  fileWrite:     writeTool,
  fileEdit:      editTool,
  fileTree:      treeTool,
  fileSearch:    searchTool,
  shell:         shellTool,
  webSearch:     webSearchTool,
  webFetch:      webFetchTool,
  artifactRead:  artifactReadTool,
  artifactWrite: artifactWriteTool,

  think:         thinkTool,
})

export const createDebugger = (systemPrompt: string) => defineAgent<typeof tools, PolicyContext>(tools, {
  id: 'debugger',
  model: 'secondary',
  systemPrompt,
  thinkingLenses: [hypothesisLens, skepticismLens, strategyLens, turnLens],
  observables: [backgroundProcessesObservable],

  permission: (p) => ({
    shell(input, ctx) {
      const result = classifyShellCommand(input.command)
      if (result.tier === 'forbidden') return p.reject(result.reason ? `This command is forbidden: ${result.reason}` : 'This command is forbidden.')
      if (detectsOutsideCwd(input.command, ctx.cwd)) return p.reject('This command targets paths outside the working directory.')
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
