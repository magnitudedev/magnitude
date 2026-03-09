export interface ThinkingLens {
  name: string
  trigger: string
  description: string
}

export function defineThinkingLens(lens: ThinkingLens): ThinkingLens {
  return lens
}

export const approvalThinkingLens = defineThinkingLens({
  name: 'approval',
  trigger: 'Before initiating any observable changes',
  description: 'Consider whether you have approval to proceed or need to check with the user before acting.',
})

export const assumptionsThinkingLens = defineThinkingLens({
  name: 'assumptions',
  trigger: 'When resolving ambiguity or making decisions the user hasn\'t explicitly specified',
  description: 'Identify what you are assuming and whether those assumptions should be communicated to the user.',
})

export const intentThinkingLens = defineThinkingLens({
  name: 'intent',
  trigger: 'When you see a user message',
  description: 'Carefully consider what the user means and what they want you to do.',
})

export const taskThinkingLens = defineThinkingLens({
  name: 'task',
  trigger: 'When current work requires problem solving or reasoning',
  description: `Think about the task at hand.
Do not think about the task for too long - prefer to ground yourself by observing and interacting with the environment instead of deliberating internally.`,
})

export const turnThinkingLens = defineThinkingLens({
  name: 'turn',
  trigger: 'When your turn involves communications and actions that could benefit from planning.',
  description: 'Briefly plan out specifically what communications you will conduct and what actions you will perform this turn.',
})

export const builtInThinkingLenses = [approvalThinkingLens, assumptionsThinkingLens, intentThinkingLens, taskThinkingLens, turnThinkingLens] as const