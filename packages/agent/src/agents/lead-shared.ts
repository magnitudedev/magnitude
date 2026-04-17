import { observe, idle, defineThinkingLens } from '@magnitudedev/roles'
import { homedir } from 'node:os'
import { join } from 'node:path'
import type { TurnPolicy } from '@magnitudedev/roles'
import type { PolicyContext } from './types'
import { catalog } from '../catalog'
import { denyForbiddenCommands, denyMassDestructiveIn, denyMutatingGit, denyWritesOutside, allowAll } from './policy'

export const alignmentLens = defineThinkingLens({
  name: 'alignment',
  trigger: 'When acting on user intent or making decisions that affect direction',
  description: 'What assumptions am I making? What decisions am I about to make that the user might want input on? Should I surface my reasoning or ask before proceeding?',
})

export const tasksLens = defineThinkingLens({
  name: 'tasks',
  trigger: 'When receiving a user message or a worker message',
  description: 'Consider task one-turnability, whether all non one-turnable work is being represented as tasks, and whether existing tasks are organized and up to date',
})

export const diligenceLens = defineThinkingLens({
  name: 'diligence',
  trigger: 'When receiving worker output or evaluating whether work is ready to present',
  description: 'Is the work my workers produced actually meeting the bar? Am I confident enough to own this output? What would I push back on?',
})

export const skillsLens = defineThinkingLens({
  name: 'skills',
  trigger: 'When planning work, creating tasks, receiving worker output, or completing work',
  description: 'Is there a skill that applies to this work? If so, have I activated it and read its guidance? What does the skill say about how to approach this, what context to share with workers, and what quality bar to meet? If work is in progress, is it meeting the skill\'s standard?',
})

export const pivotLens = defineThinkingLens({
  name: 'pivot',
  trigger: 'When progress has stalled or results are unexpected',
  description: 'Is the current approach working? Should I change direction, re-scope, or escalate?',
})

export const turnLens = defineThinkingLens({
  name: 'turn',
  trigger: 'When the turn involves multiple messages/tools',
  description: 'What to communicate, what actions to take, which turn control to use.',
})

export const constraintsLens = defineThinkingLens({
  name: 'constraints',
  trigger: 'When planning work, delegating to subagents, or evaluating progress',
  description:
    'What are the exact requirements? Have I extracted all testable constraints? Which have I verified? Which remain? Am I missing any implicit requirements?',
})

export const validationLens = defineThinkingLens({
  name: 'validation',
  trigger: 'When evaluating whether work is complete or results are acceptable',
  description:
    'Have I empirically tested my complete solution, not just individual pieces? Are there edge cases or details I haven\'t checked? Am I accepting results that look wrong or suspicious?',
})

export const leadTools = catalog.pick(
  'fileRead',
  'fileWrite',
  'fileEdit',
  'fileTree',
  'fileSearch',
  'fileView',
  'shell',
  'webFetch',
  'webSearch',
  'agentCreate',
  'agentKill',
  'createTask',
  'updateTask',
  'spawnWorker',
  'killWorker',
  'skill',
)

export const leadObservables: readonly [] = []

export const leadPolicy = [
  denyForbiddenCommands(),
  denyMutatingGit(),
  denyWritesOutside((ctx: PolicyContext) => [ctx.cwd, ctx.workspacePath, join(homedir(), '.magnitude')]),
  denyMassDestructiveIn(() => [join(homedir(), '.magnitude')]),
  allowAll(),
]

export const leadTurnPolicy: TurnPolicy<typeof leadTools, PolicyContext> = {
  decide(turnCtx) {
    if (turnCtx.cancelled) return idle()
    if (turnCtx.error) return observe()
    if (turnCtx.toolsCalled.length === 0) return idle()

    if (turnCtx.messagesSent.some((m: { taskId: string | null }) => m.taskId !== null)) return idle()
    return observe()
  },
}
