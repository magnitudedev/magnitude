import type { ExecuteHookContext, InterceptorDecision } from '@magnitudedev/harness'
import type { Effect } from 'effect'
import type { PromptTemplate } from './prompt'

export type Slot = 'leader' | 'scout' | 'architect' | 'engineer' | 'critic' | 'scientist' | 'artisan' | 'advisor'

export const SLOTS: readonly Slot[] = ['leader', 'scout', 'architect', 'engineer', 'critic', 'scientist', 'artisan', 'advisor'] as const

export function isSlot(value: string): value is Slot {
  return (SLOTS as readonly string[]).includes(value)
}

export interface PolicyContext {
  readonly cwd: string
  readonly workspacePath: string
  readonly disableShellSafeguards?: boolean
  readonly disableCwdSafeguards?: boolean
}

export type PolicyRule = (ctx: ExecuteHookContext & { policyContext: PolicyContext }) =>
  Effect.Effect<InterceptorDecision | null>

export interface RoleDefinition {
  /** Role identity — same as the slot it fills. */
  readonly id: Slot

  /** System prompt template. Render with runtime vars (e.g. SKILLS_SECTION) to get final text. */
  readonly prompt: PromptTemplate<'SKILLS_SECTION'>

  /** Where messages go by default ('user' for lead, 'parent' for subagents). */
  readonly defaultRecipient: 'user' | 'parent'

  /** Protocol mode — affects prompt structure and available agent tools. */
  readonly protocolRole: 'lead' | 'subagent'

  /** Whether this role can be spawned as a worker by the lead. */
  readonly spawnable: boolean

  /** Policy rules — evaluated by the agent to build a beforeExecute hook. */
  readonly policy: PolicyRule[]

  /** Lifecycle prompts shown to parent/user at spawn/completion. */
  readonly lifecycle?: {
    readonly parentOnSpawn?: string
    readonly parentOnIdle?: string
  }

  /** Initial context flags. */
  readonly initialContext: {
    readonly parentConversation: boolean
  }
}
