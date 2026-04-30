import { definePrompt } from '../prompt'
import { denyForbiddenCommands, denyMutatingGit, denyWritesOutside, denyMassDestructiveIn, allowAll } from '../policy'
import type { RoleDefinition } from '../types'
import scoutPromptRaw from '../prompts/scout.txt' with { type: 'text' }
import { homedir } from 'node:os'
import { join } from 'node:path'

export function createScoutRole(): RoleDefinition {
  return {
    id: 'scout',
    prompt: definePrompt<'SKILLS_SECTION'>(scoutPromptRaw),
    defaultRecipient: 'parent',
    protocolRole: 'subagent',
    spawnable: true,
    policy: [
      denyForbiddenCommands(),
      denyMutatingGit(),
      denyWritesOutside(ctx => [ctx.cwd, ctx.workspacePath, join(homedir(), '.magnitude')]),
      denyMassDestructiveIn(ctx => [join(homedir(), '.magnitude')]),
      allowAll(),
    ],
    lifecycle: {
      parentOnSpawn: 'If there are other areas to investigate, spawn additional scouts in parallel.',
      parentOnIdle: "Review the scout's findings for relevance and completeness.",
    },
    initialContext: { parentConversation: true },
  }
}
