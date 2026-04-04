import { continue_, yield_, defineThinkingLens } from '@magnitudedev/roles'
import { homedir } from 'node:os'
import { join } from 'node:path'
import type { TurnPolicy } from '@magnitudedev/roles'
import type { PolicyContext } from './types'
import { catalog } from '../catalog'
import { denyForbiddenCommands, denyMassDestructiveIn, denyMutatingGit, denyWritesOutside, allowAll } from './policy'

export const intentLens = defineThinkingLens({
  name: 'intent',
  trigger: 'When you receive a message from the user',
  description: 'Carefully consider what the user means and what they actually want. Look past the literal request to understand the underlying goal.',
})

export const ideateLens = defineThinkingLens({
  name: 'ideate',
  trigger: 'When the problem requires creative thinking or there are multiple possible approaches',
  description: 'Think freely about the problem space. Generate and consider different approaches, ideas, or solutions before committing to one. Explore tradeoffs and implications.',
})

export const strategyLens = defineThinkingLens({
  name: 'strategy',
  trigger: 'When deciding how to execute work',
  description: 'Consider how to best tackle work - subagents, parallelism, sequencing, workspace usage.',
})

export const traitsLens = defineThinkingLens({
  name: 'traits',
  trigger: 'Whenever one or more traits are applicable to the current situation',
  description: 'Assess how one or traits might apply in the current situation',
})

export const workflowLens = defineThinkingLens({
  name: 'workflow',
  trigger: 'When deciding how to tackle work',
  description: 'How should I execute this? Which subagents, what parallelism, what sequencing, what workspace usage?',
})

export const constraintsLens = defineThinkingLens({
  name: 'constraints',
  trigger: 'When planning work, delegating to subagents, or evaluating progress',
  description:
    'What are the exact requirements? Have I extracted all testable constraints? Which have I verified? Which remain? Am I missing any implicit requirements?',
})

export const pivotLens = defineThinkingLens({
  name: 'pivot',
  trigger: 'When an approach is not making progress or results are unexpected',
  description:
    'Is my current approach working? Are my subagents stuck or spinning? Should I try a different strategy, parallelize an alternative, or cut losses on this path? What signals indicate I should change direction?',
})

export const validationLens = defineThinkingLens({
  name: 'validation',
  trigger: 'When evaluating whether work is complete or results are acceptable',
  description:
    'Have I empirically tested my complete solution, not just individual pieces? Are there edge cases or details I haven\'t checked? Am I accepting results that look wrong or suspicious?',
})

export const turnLens = defineThinkingLens({
  name: 'turn',
  trigger: 'When your turn involves communications and actions that could benefit from planning',
  description: 'Plan what to communicate, what actions to take, and which turn control to use. If acting this turn, remember that you cannot communicate the results of those actions until next turn.',
})

export const taskLens = defineThinkingLens({
  name: 'task',
  trigger: 'When receiving user request or performing work of any kind',
  description:
    "Is all work captured as tasks? Am I about to do something that should be a task but isn't? Are there implicit subtasks I haven't created yet? Every piece of work — no matter how small — must be represented as a task.",
})

export const leadTools = catalog.pick(
  'fileRead',
  'fileWrite',
  'fileEdit',
  'fileTree',
  'fileSearch',
  'fileView',
  'shell',
  'webSearch',
  'webFetch',

  'skill',
  'phaseSubmit',
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
    if (turnCtx.cancelled) return yield_()
    if (turnCtx.error) return continue_()
    if (turnCtx.toolsCalled.length === 0) return yield_()

    if (turnCtx.messagesSent.some((m: { taskId: string | null }) => m.taskId !== null)) return yield_()
    return continue_()
  },
}
