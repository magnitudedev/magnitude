/**
 * Reviewer Agent Definition
 *
 * Independently verifies implemented changes meet the user's intent.
 * Read-only codebase access + shell for running tests/builds.
 * No file write access — reviewers only observe, test, and report.
 */

import { toolSet, defineRole, continue_, yield_, finish, defineThinkingLens } from '@magnitudedev/roles'
import reviewerPromptRaw from './prompts/reviewer.txt' with { type: 'text' }
import { compilePromptTemplate } from '../prompts/system-prompt'
import { readTool, treeTool, searchTool, viewTool } from '../tools/fs'
import { shellBgTool } from '../tools/shell-bg'
import { shellTool } from '../tools/shell'


import { agentCreateTool } from '../tools/agent-tools'
import { phaseVerdictTool } from '../tools/phase-verdict'
import { classifyShellCommand, writesStayWithin } from '@magnitudedev/shell-classifier'
import type { PolicyContext } from './types'
import { backgroundProcessesObservable } from '../observables/background-processes-observable'

const intentLens = defineThinkingLens({
  name: 'intent',
  trigger: 'When beginning review or evaluating changes',
  description: "What did the user actually ask for? Re-read the original request and any plans. Evaluate the work against the user's intent, not just against whether the code looks reasonable.",
})

const qualityLens = defineThinkingLens({
  name: 'quality',
  trigger: 'When examining implemented code',
  description: 'Does the implementation match existing patterns and conventions? Is it consistent with the surrounding codebase? Look for style mismatches, abstraction violations, and unnecessary complexity.',
})

const skepticismLens = defineThinkingLens({
  name: 'skepticism',
  trigger: 'When evaluating whether work is complete and correct',
  description: "Assume nothing works until proven. What could still be wrong? What edge cases haven't been tested? What claims are being made without evidence? Don't accept code reading as proof of correctness — run things.",
})

const turnLens = defineThinkingLens({
  name: 'turn',
  trigger: 'When planning your next actions',
  description: 'Plan what to verify this turn. What tests to run, what commands to execute, what code to inspect? Prioritize execution-based verification over code reading.',
})

const systemPrompt = compilePromptTemplate(reviewerPromptRaw)

const tools = toolSet({
  fileRead:       readTool,
  fileTree:       treeTool,
  fileSearch:     searchTool,
  fileView:       viewTool,
  shell:          shellTool,
  shellBg:        shellBgTool,

  agentCreate:    agentCreateTool,

  phaseVerdict:    phaseVerdictTool,
})

export const reviewerRole = defineRole<typeof tools, 'secondary', PolicyContext>({
  tools,
  id: 'reviewer',
  slot: 'secondary',
  systemPrompt,
  lenses: [intentLens, qualityLens, skepticismLens, turnLens],
  defaultRecipient: 'parent',
  protocolRole: 'subagent',
  initialContext: { parentConversation: true },
  spawnable: true,
  observables: [backgroundProcessesObservable],

  permission: (p) => ({
    shell(input: { command: string }, ctx) {
      const result = classifyShellCommand(input.command)
      const allowedPrefixes = ctx.workspacePath ? [ctx.workspacePath] : undefined
      if (result.tier === 'forbidden') return p.reject(result.reason ? `This command is forbidden: ${result.reason}` : 'This command is forbidden.')
      if (!writesStayWithin(input.command, ctx.cwd, ...(allowedPrefixes ?? []))) return p.reject('This command targets paths outside the working directory.')
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
})
