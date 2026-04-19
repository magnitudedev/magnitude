/**
 * Builder Agent Definition
 *
 * Full read/write agent that implements tasks. Has fs, shell, web search,
 * and parent.message for communicating with the orchestrator. No task management tools.
 */

import { defineRole, observe, idle, defineThinkingLens } from '@magnitudedev/roles'
import { homedir } from 'node:os'
import { join } from 'node:path'
import builderPromptRaw from './prompts/builder.txt' with { type: 'text' }
import { compilePromptTemplate } from '../prompts/system-prompt'
import { catalog } from '../catalog'
import { denyForbiddenCommands, denyMassDestructiveIn, denyMutatingGit, denyWritesOutside, allowAll } from './policy'
import type { PolicyContext } from './types'
import { formatAgentIdList } from './lifecycle-reminder-format'


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

const systemPrompt = compilePromptTemplate(builderPromptRaw)

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

export const builderRole = defineRole<typeof tools, 'builder', PolicyContext>({
  tools,
  id: 'builder',
  slot: 'builder',
  systemPrompt,
  lenses: [qualityLens, turnLens],
  defaultRecipient: 'parent',
  protocolRole: 'subagent',
  initialContext: { parentConversation: true },
  spawnable: true,
  observables: [],
  lifecyclePrompts: {
    parentOnSpawn: (agentIds) =>
      `If there are other independent changes to make, spawn additional builders in parallel rather than waiting for ${formatAgentIdList(agentIds)}.`,
    parentOnIdle: (agentIds) =>
      `Review ${formatAgentIdList(agentIds)}'s work for correctness, quality, and adherance to user requirements. For nontrivial changes, spawn a reviewer. Do not present unverified work to user.`,
  },

  policy: [
    denyForbiddenCommands(),
    denyMutatingGit(),
    denyWritesOutside(ctx => [ctx.cwd, ctx.workspacePath, join(homedir(), '.magnitude')]),
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
