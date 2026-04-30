import { definePrompt } from '../prompt'
import { denyForbiddenCommands, denyMutatingGit, denyWritesOutside, denyMassDestructiveIn, allowAll } from '../policy'
import type { RoleDefinition } from '../types'
import scientistPromptRaw from '../prompts/scientist.txt' with { type: 'text' }
import { homedir } from 'node:os'
import { join } from 'node:path'

export function createScientistRole(): RoleDefinition {
  return {
    id: 'scientist',
    prompt: definePrompt<'SKILLS_SECTION'>(scientistPromptRaw),
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
      parentOnSpawn: undefined,
      parentOnIdle: "Review the scientist's diagnosis and determine next steps.",
    },
    initialContext: { parentConversation: true },
  }
}
