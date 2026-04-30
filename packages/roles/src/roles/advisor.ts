import { definePrompt } from '../prompt'
import { denyForbiddenCommands, denyMutatingGit, denyWritesOutside, denyMassDestructiveIn, allowAll } from '../policy'
import type { RoleDefinition } from '../types'
import advisorPromptRaw from '../prompts/advisor.txt' with { type: 'text' }
import { homedir } from 'node:os'
import { join } from 'node:path'

export function createAdvisorRole(): RoleDefinition {
  return {
    id: 'advisor',
    prompt: definePrompt<'SKILLS_SECTION'>(advisorPromptRaw),
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
      parentOnIdle: "Review the advisor's recommendations.",
    },
    initialContext: { parentConversation: true },
  }
}
