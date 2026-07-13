import { definePrompt } from '../prompt'
import { denyForbiddenCommands, denyMutatingGit, denyWritesOutside, denyMassDestructiveIn, allowAll } from '../policy'
import type { RoleDefinition } from '../types'
import artisanPromptRaw from '../prompts/artisan.txt' with { type: 'text' }
import { homedir } from 'node:os'
import { join } from 'node:path'

export function createArtisanRole(): RoleDefinition {
  return {
    id: 'artisan',
    description: 'Crafts specialized implementations',
    prompt: definePrompt<'SKILLS_SECTION' | 'THINKING_LIMIT' | 'CHECKPOINT_SECTION'>(artisanPromptRaw),
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
      coordinatorOnSpawn: 'If there are other artifacts to create, spawn additional artisans in parallel.',
      coordinatorOnIdle: "Review the artisan's output for quality and completeness.",
    },
    initialContext: { coordinatorConversation: true },
  }
}
