import { definePrompt } from '../prompt'
import { denyForbiddenCommands, denyMutatingGit, denyWritesOutside, denyMassDestructiveIn, allowAll } from '../policy'
import type { RoleDefinition } from '../types'
import architectPromptRaw from '../prompts/architect.txt' with { type: 'text' }
import { homedir } from 'node:os'
import { join } from 'node:path'

export function createArchitectRole(): RoleDefinition {
  return {
    id: 'architect',
    description: 'Plans structure and design',
    prompt: definePrompt<'SKILLS_SECTION' | 'THINKING_LIMIT' | 'CHECKPOINT_SECTION'>(architectPromptRaw),
    defaultRecipient: 'coordinator',
    agentKind: 'worker',
    spawnable: true,
    maxThoughtChars: 20000,
    policy: [
      denyForbiddenCommands(),
      denyMutatingGit(),
      denyWritesOutside(ctx => [ctx.cwd, ctx.scratchpadPath, join(homedir(), '.magnitude')]),
      denyMassDestructiveIn(ctx => [join(homedir(), '.magnitude')]),
      allowAll(),
    ],
    lifecycle: {
      coordinatorOnSpawn: undefined,
      coordinatorOnIdle: "Review the architect's plan for completeness and alignment with requirements.",
    },
    initialContext: { coordinatorConversation: true },
  }
}
