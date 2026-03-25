/**
 * Debugger Agent Definition
 *
 * Root-cause investigation agent with full read/write + shell access.
 * Focuses on diagnosis — forming hypotheses, testing them, narrowing down causes.
 */

import { toolSet, defineRole, continue_, yield_, finish, defineThinkingLens } from '@magnitudedev/roles'
import debuggerPromptRaw from './prompts/debugger.txt' with { type: 'text' }
import { compilePromptTemplate } from '../prompts/system-prompt'
import { readTool, writeTool, editTool, treeTool, searchTool, viewTool } from '../tools/fs'
import { shellBgTool } from '../tools/shell-bg'
import { shellTool } from '../tools/shell'
import { webSearchTool } from '../tools/web-search-tool'
import { webFetchTool } from '../tools/web-fetch-tool'

import { denyForbiddenCommands, denyMutatingGit, denyWritesOutside, allowAll } from './policy'
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

const systemPrompt = compilePromptTemplate(debuggerPromptRaw)

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

export const debuggerRole = defineRole<typeof tools, 'debugger', PolicyContext>({
  tools,
  id: 'debugger',
  slot: 'debugger',
  systemPrompt,
  lenses: [hypothesisLens, skepticismLens, strategyLens, turnLens],
  defaultRecipient: 'parent',
  protocolRole: 'subagent',
  initialContext: { parentConversation: true },
  spawnable: true,
  observables: [backgroundProcessesObservable],

  policy: [
    denyForbiddenCommands(),
    denyMutatingGit(),
    denyWritesOutside(ctx => [ctx.cwd, ctx.workspacePath]),
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
