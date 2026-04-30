import { definePrompt } from '../prompt'
import { denyForbiddenCommands, denyMutatingGit, denyWritesOutside, denyMassDestructiveIn, allowAll } from '../policy'
import type { RoleDefinition } from '../types'
import engineerPromptRaw from '../prompts/engineer.txt' with { type: 'text' }
import { homedir } from 'node:os'
import { join } from 'node:path'

export function createEngineerRole(): RoleDefinition {
  return {
    id: 'engineer',
    prompt: definePrompt<'SKILLS_SECTION'>(engineerPromptRaw),
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
      parentOnSpawn: 'If there are other independent changes to make, spawn additional engineers in parallel.',
      parentOnIdle: "Review the engineer's work for correctness and quality.",
    },
    initialContext: { parentConversation: true },
  }
}
