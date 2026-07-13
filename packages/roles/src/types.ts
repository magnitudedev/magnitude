import type { ExecuteHookContext, InterceptorDecision } from '@magnitudedev/harness'
import type { Effect } from 'effect'
import type { PromptTemplate } from './prompt'
import type { SlotId, RoleId } from './constants'

// Pure constants and types re-exported from ./constants (zero runtime deps)
export type { SlotId, RoleId } from './constants'
export {
  ROLE_IDS,
  isRoleId,
  ROLE_TO_SLOT,
  DEFAULT_REASONING_EFFORT,
  SLOT_IDS,
  SLOT_DISPLAY_NAMES,
  SLOT_DESCRIPTIONS,
} from './constants'

export interface PolicyContext {
  readonly cwd: string
  readonly scratchpadPath: string
  readonly disableShellSafeguards?: boolean
  readonly disableCwdSafeguards?: boolean
}

export type PolicyRule = (ctx: ExecuteHookContext & { policyContext: PolicyContext }) =>
  Effect.Effect<InterceptorDecision | null>

export interface RoleDefinition {
  /** Role identity. */
  readonly id: RoleId

  /** Short human-readable description of this role. */
  readonly description: string

  /** System prompt template. Render with runtime vars (e.g. SKILLS_SECTION, THINKING_LIMIT, CHECKPOINT_SECTION) to get final text. */
  readonly prompt: PromptTemplate<'SKILLS_SECTION' | 'THINKING_LIMIT' | 'CHECKPOINT_SECTION'>

  /** Where messages go by default ('user' for lead, 'coordinator' for workers). */
  readonly defaultRecipient: 'user' | 'coordinator'

  /** Agent kind — lead, worker, or peer. Affects prompt structure and available agent tools. */
  readonly agentKind: 'lead' | 'worker' | 'peer'

  /** Whether this role can be spawned as a worker by the lead. */
  readonly spawnable: boolean

  /** Policy rules — evaluated by the agent to build a beforeExecute hook. */
  readonly policy: PolicyRule[]

  /** Lifecycle prompts shown to coordinator/user at spawn/completion. */
  readonly lifecycle?: {
    readonly coordinatorOnSpawn?: string
    readonly coordinatorOnIdle?: string
  }

  /** Initial context flags. */
  readonly initialContext: {
    readonly coordinatorConversation: boolean
  }

  /** Maximum characters allowed in a single thinking block before mechanical abort. */
  readonly maxThoughtChars?: number
}
