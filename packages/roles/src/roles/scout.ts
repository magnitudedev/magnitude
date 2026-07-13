import { definePrompt } from '../prompt'
import { denyForbiddenCommands, denyMutatingGit, denyWritesOutside, denyMassDestructiveIn, allowAll } from '../policy'
import type { RoleDefinition } from '../types'
import scoutPromptRaw from '../prompts/scout.txt' with { type: 'text' }
import { homedir } from 'node:os'
import { join } from 'node:path'

export function createScoutRole(): RoleDefinition {
  return {
    id: 'scout',
    description: 'Explores and investigates codebase',
    prompt: definePrompt<'SKILLS_SECTION' | 'THINKING_LIMIT' | 'CHECKPOINT_SECTION'>(scoutPromptRaw),
    defaultRecipient: 'coordinator',
    agentKind: 'worker',
    spawnable: true,
    maxThoughtChars: 2000,
    policy: [
      denyForbiddenCommands(),
      denyMutatingGit(),
      denyWritesOutside(ctx => [ctx.cwd, ctx.scratchpadPath, join(homedir(), '.magnitude')]),
      denyMassDestructiveIn(ctx => [join(homedir(), '.magnitude')]),
      allowAll(),
    ],
    lifecycle: {
      coordinatorOnSpawn: 'If there are other areas to investigate, spawn additional scouts in parallel.',
      coordinatorOnIdle: "Review the scout's findings for relevance and completeness.",
    },
    initialContext: { coordinatorConversation: true },
  }
}
