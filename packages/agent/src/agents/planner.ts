/**
 * Planner Agent Definition
 *
 * Agent that produces implementation plans and makes decisions.
 * Has read-only shell access, but can write files within the workspace.
 * Uses primary model (planning needs reasoning power).
 * Communicates back via parent.message.
 */

import { defineRole, continue_, yield_, finish, defineThinkingLens } from '@magnitudedev/roles'
import plannerPromptRaw from './prompts/planner.txt' with { type: 'text' }
import { compilePromptTemplate } from '../prompts/system-prompt'
import { catalog } from '../catalog'

// import { gatherTool } from '../tools/gather'
import { allowReadonlyShell, denyForbiddenCommands, denyMutatingGit, denyWritesOutside, allowAll } from './policy'
import type { PolicyContext } from './types'
import { formatAgentIdList } from './lifecycle-reminder-format'


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

const systemPrompt = compilePromptTemplate(plannerPromptRaw)

const tools = catalog.pick(
  'fileRead',
  'fileWrite',
  'fileEdit',
  'fileTree',
  'fileSearch',
  'shell',
  'webSearch',
  'webFetch',
  // 'gather',
)

export const plannerRole = defineRole<typeof tools, 'planner', PolicyContext>({
  tools,
  id: 'planner',
  slot: 'planner',
  systemPrompt,
  lenses: [ideateLens, velocityLens, alignmentLens, capacityLens, turnLens],
  defaultRecipient: 'parent',
  protocolRole: 'subagent',
  initialContext: { parentConversation: true },
  spawnable: true,
  observables: [],
  lifecyclePrompts: {
    parentOnSpawn: (agentIds) =>
      `If the task has multiple large independent facets that need separate plans, spawn additional planners in parallel rather than waiting for ${formatAgentIdList(agentIds)}.`,
    parentOnIdle: (agentIds) =>
      `Critique ${formatAgentIdList(agentIds)}'s plan against the user's stated requirements and intent. If the plan is not solid, message the planner with specific feedback to revise. Do not present the plan to the user or proceed to building until the plan is concrete, complete, and aligned with the user's intent.`,
  },

  policy: [
    allowReadonlyShell(),
    denyForbiddenCommands(),
    denyMutatingGit(),
    denyWritesOutside(ctx => [ctx.workspacePath]),
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
