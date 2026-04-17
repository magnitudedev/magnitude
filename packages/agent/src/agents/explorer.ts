/**
 * Explorer Agent Definition
 *
 * Agent that answers informational questions by exploring the codebase.
 * Has read-only shell access, but can write files within the workspace.
 * Uses secondary model. Communicates back via parent.message.
 */

import { defineRole, observe, idle, finish, defineThinkingLens } from '@magnitudedev/roles'
import { homedir } from 'node:os'
import { join } from 'node:path'
import explorerPromptRaw from './prompts/explorer.txt' with { type: 'text' }
import { compilePromptTemplate } from '../prompts/system-prompt'
import { catalog } from '../catalog'

import { allowReadonlyShell, denyForbiddenCommands, denyMassDestructiveIn, denyMutatingGit, denyWritesOutside, allowAll } from './policy'
import type { PolicyContext } from './types'
import { formatAgentIdList } from './lifecycle-reminder-format'


const strategyLens = defineThinkingLens({
  name: 'strategy',
  trigger: 'On new session, or on new direction from parent',
  description: "Consider overall approach",
})

const turnLens = defineThinkingLens({
  name: 'turn',
  trigger: 'When taking making one or more tool calls',
  description: 'How to structure this specific response, what tools and/or messages to use and how',
})

const systemPrompt = compilePromptTemplate(explorerPromptRaw)

const tools = catalog.pick(
  'fileRead',
  'fileWrite',
  'fileEdit',
  'fileTree',
  'fileSearch',
  'fileView',
  'shell',
  'webFetch',
  'skill',
)

export const explorerRole = defineRole<typeof tools, 'explorer', PolicyContext>({
  tools,
  id: 'explorer',
  slot: 'explorer',
  systemPrompt,
  lenses: [strategyLens, turnLens],
  defaultRecipient: 'parent',
  protocolRole: 'subagent',
  initialContext: { parentConversation: true },
  spawnable: true,
  observables: [],
  lifecyclePrompts: {
    parentOnSpawn: (agentIds) =>
      `If you need context on multiple areas, spawn additional explorers in parallel rather than waiting for ${formatAgentIdList(agentIds)}.`,
    parentOnIdle: (agentIds) =>
      `Evaluate whether the findings of ${formatAgentIdList(agentIds)} are sufficient. If ambiguities or unknowns remain, send ${agentIds.length === 1 ? agentIds[0] : 'them'} back with specific questions or spawn additional explorers. Do not proceed to planning or building with incomplete context.`,
  },

  policy: [
    allowReadonlyShell(),
    denyForbiddenCommands(),
    denyMutatingGit(),
    denyWritesOutside(ctx => [ctx.workspacePath, join(homedir(), '.magnitude')]),
    denyMassDestructiveIn(() => [join(homedir(), '.magnitude')]),
    allowAll(),
  ],

  turn: {
    decide(turnCtx) {
      if (turnCtx.cancelled) return finish()
      if (turnCtx.error) return observe()
      if (turnCtx.toolsCalled.length === 0 && turnCtx.messagesSent.some(m => m.taskId === null)) return idle()
      return observe()
    },
  },
})