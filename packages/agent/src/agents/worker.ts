/**
 * Generic Worker Agent Definition
 *
 * A single generic worker that handles all work types. Full read/write access.
 * Workers can activate skills via the `<skill>` tool when they need methodology guidance.
 */

import { defineRole, observe, idle, defineThinkingLens } from '@magnitudedev/roles'
import { homedir } from 'node:os'
import { join } from 'node:path'
import workerPromptRaw from './prompts/worker.txt' with { type: 'text' }
import { compilePromptTemplate } from '../prompts/system-prompt'
import { catalog } from '../catalog'
import { denyForbiddenCommands, denyMutatingGit, denyWritesOutside, denyMassDestructiveIn, allowAll } from './policy'
import type { PolicyContext } from './types'

const turnLens = defineThinkingLens({
  name: 'turn',
  trigger: 'When planning your next actions',
  description: "Plan what to read and edit this turn. What files do you need to understand before making changes? What's the right order of edits?",
})

const skillsLens = defineThinkingLens({
  name: 'skills',
  trigger: 'When starting new work or uncertain about approach',
  description: 'Should I activate a skill for methodology guidance? The `<skill>` tool loads detailed approaches for specific work types (research, plan, implement, debug, etc.). Activate early when the work type is clear.',
})

const systemPrompt = compilePromptTemplate(workerPromptRaw)

const tools = catalog.pick(
  'fileRead',
  'fileWrite',
  'fileEdit',
  'fileTree',
  'fileSearch',
  'fileView',
  'shell',
  'webFetch',
  'webSearch',
  'skill',
)

export const workerRole = defineRole<typeof tools, 'worker', PolicyContext>({
  tools,
  id: 'worker',
  slot: 'worker',
  systemPrompt,
  lenses: [turnLens, skillsLens],
  defaultRecipient: 'parent',
  protocolRole: 'subagent',
  initialContext: { parentConversation: true },
  spawnable: true,
  observables: [],

  policy: [
    denyForbiddenCommands(),
    denyMutatingGit(),
    denyWritesOutside((ctx: PolicyContext) => [ctx.cwd, ctx.workspacePath, join(homedir(), '.magnitude')]),
    denyMassDestructiveIn(() => [join(homedir(), '.magnitude')]),
    allowAll(),
  ],

  turn: {
    decide(turnCtx) {
      if (turnCtx.cancelled) return idle()
      if (turnCtx.error) return observe()
      if (turnCtx.toolsCalled.length === 0 && turnCtx.messagesSent.some(m => m.taskId === null)) return idle()
      return observe()
    },
  },
})
