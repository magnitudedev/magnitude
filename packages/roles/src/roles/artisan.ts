import { definePrompt } from '../prompt'
import { denyForbiddenCommands, denyMutatingGit, denyWritesOutside, denyMassDestructiveIn, allowAll } from '../policy'
import type { RoleDefinition } from '../types'
import artisanPromptRaw from '../prompts/artisan.txt' with { type: 'text' }
import { homedir } from 'node:os'
import { join } from 'node:path'

export function createArtisanRole(): RoleDefinition {
  return {
    id: 'artisan',
    prompt: definePrompt<'SKILLS_SECTION'>(artisanPromptRaw),
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
      parentOnSpawn: 'If there are other artifacts to create, spawn additional artisans in parallel.',
      parentOnIdle: "Review the artisan's output for quality and completeness.",
    },
    initialContext: { parentConversation: true },
  }
}
