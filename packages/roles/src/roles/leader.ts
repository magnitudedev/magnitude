import { definePrompt } from '../prompt'
import { denyForbiddenCommands, denyMutatingGit, denyWritesOutside, denyMassDestructiveIn, allowAll } from '../policy'
import type { RoleDefinition } from '../types'
import leaderPromptRaw from '../prompts/leader.txt' with { type: 'text' }
import { homedir } from 'node:os'
import { join } from 'node:path'

export function createLeaderRole(): RoleDefinition {
  return {
    id: 'leader',
    prompt: definePrompt<'SKILLS_SECTION'>(leaderPromptRaw),
    defaultRecipient: 'user',
    protocolRole: 'lead',
    spawnable: false,
    policy: [
      denyForbiddenCommands(),
      denyMutatingGit(),
      denyWritesOutside(ctx => [ctx.cwd, ctx.workspacePath, join(homedir(), '.magnitude')]),
      denyMassDestructiveIn(ctx => [join(homedir(), '.magnitude')]),
      allowAll(),
    ],
    lifecycle: undefined,
    initialContext: { parentConversation: false },
  }
}
