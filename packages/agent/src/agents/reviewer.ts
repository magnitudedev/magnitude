/**
 * Reviewer Agent Definition
 *
 * Independently verifies implemented changes meet the user's intent.
 * Codebase access + shell for running tests/builds.
 * Can write within workspace for notes/reports; cannot write project files.
 */

import { defineRole, continue_, yield_, finish, defineThinkingLens } from '@magnitudedev/roles'
import { homedir } from 'node:os'
import { join } from 'node:path'
import reviewerPromptRaw from './prompts/reviewer.txt' with { type: 'text' }
import { compilePromptTemplate } from '../prompts/system-prompt'
import { catalog } from '../catalog'
import { denyForbiddenCommands, denyMassDestructiveIn, denyMutatingGit, denyWritesOutside, allowAll } from './policy'
import type { PolicyContext } from './types'
import { formatAgentIdList } from './lifecycle-reminder-format'


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

const tools = catalog.pick(
  'fileRead',
  'fileWrite',
  'fileEdit',
  'fileTree',
  'fileSearch',
  'fileView',
  'shell',
  'assignTask',
  'phaseVerdict',
)

export const reviewerRole = defineRole<typeof tools, 'reviewer', PolicyContext>({
  tools,
  id: 'reviewer',
  slot: 'reviewer',
  systemPrompt,
  lenses: [intentLens, qualityLens, skepticismLens, turnLens],
  defaultRecipient: 'parent',
  protocolRole: 'subagent',
  initialContext: { parentConversation: true },
  spawnable: true,
  observables: [],
  lifecyclePrompts: {
    parentOnIdle: (agentIds) =>
      `Address ALL findings from ${formatAgentIdList(agentIds)}. Have builders fix the issues and run another review cycle. Do not stop working or report to the user until all findings are resolved and the work is correct, high quality, and fully meets the user's requirements.`,
  },

  policy: [
    denyForbiddenCommands(),
    denyMutatingGit(),
    denyWritesOutside(ctx => [ctx.workspacePath, join(homedir(), '.magnitude')]),
    denyMassDestructiveIn(() => [join(homedir(), '.magnitude')]),
    allowAll(),
  ],

  turn: {
    decide(turnCtx) {
      if (turnCtx.cancelled) return finish()
      if (turnCtx.error) return continue_()
      if (turnCtx.toolsCalled.length === 0 && turnCtx.messagesSent.some(m => m.dest === 'parent')) return yield_()
      if (turnCtx.toolsCalled.some(t => t === 'assignTask')) return yield_()
      return continue_()
    },
  },
})
