/**
 * Generic Worker Agent Definition
 *
 * A single generic worker that handles all task types. Full read/write access.
 * Task-specific context is provided by the lead via the skill system.
 */

import { defineRole, observe, idle, finish, defineThinkingLens } from '@magnitudedev/roles'
import { homedir } from 'node:os'
import { join } from 'node:path'
import subagentBaseRaw from './prompts/subagent-base.txt' with { type: 'text' }
import { compilePromptTemplate } from '../prompts/system-prompt'
import { catalog } from '../catalog'
import { allowAll } from './policy'
import type { PolicyContext } from './types'

const turnLens = defineThinkingLens({
  name: 'turn',
  trigger: 'When planning your next actions',
  description: "Plan what to read and edit this turn. What files do you need to understand before making changes? What's the right order of edits?",
})

const systemPrompt = compilePromptTemplate(subagentBaseRaw)

const tools = catalog.pick(
  'fileRead',
  'fileWrite',
  'fileEdit',
  'fileTree',
  'fileSearch',
  'fileView',
  'shell',
  'webSearch',
  'webFetch',
)

export const workerRole = defineRole<typeof tools, 'worker', PolicyContext>({
  tools,
  id: 'worker',
  slot: 'worker',
  systemPrompt,
  lenses: [turnLens],
  defaultRecipient: 'parent',
  protocolRole: 'subagent',
  initialContext: { parentConversation: true },
  spawnable: true,
  observables: [],

  policy: [
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
