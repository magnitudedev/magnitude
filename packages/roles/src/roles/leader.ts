import { definePrompt } from '../prompt'
import { denyForbiddenCommands, denyMutatingGit, denyWritesOutside, denyMassDestructiveIn, allowAll } from '../policy'
import type { RoleDefinition } from '../types'
import leaderPromptRaw from '../prompts/leader.txt' with { type: 'text' }
import { homedir } from 'node:os'
import { join } from 'node:path'

export function createLeaderRole(): RoleDefinition {
  return {
    id: 'leader',
    description: 'Coordinates tasks and workers',
    prompt: definePrompt<'SKILLS_SECTION' | 'THINKING_LIMIT' | 'CHECKPOINT_SECTION'>(leaderPromptRaw),
    defaultRecipient: 'user',
    agentKind: 'lead',
    spawnable: false,
    maxThoughtChars: 20000, // This is now just a fallback for if some reason grammar version doesn't trigger
    policy: [
      denyForbiddenCommands(),
      denyMutatingGit(),
      denyWritesOutside(ctx => [ctx.cwd, ctx.scratchpadPath, join(homedir(), '.magnitude')]),
      denyMassDestructiveIn(ctx => [join(homedir(), '.magnitude')]),
      allowAll(),
    ],
    lifecycle: undefined,
    initialContext: { coordinatorConversation: false },
  }
}
