// Types
export type { RoleId, SlotId, PolicyContext, PolicyRule, RoleDefinition } from './types'
export { ROLE_IDS, isRoleId, ROLE_TO_SLOT, DEFAULT_REASONING_EFFORT, SLOT_IDS, SLOT_DISPLAY_NAMES, SLOT_DESCRIPTIONS } from './types'

// Policy
export {
  denyForbiddenCommands,
  denyMutatingGit,
  denyWritesOutside,
  denyMassDestructiveIn,
  allowAll,
  evaluatePolicy,
} from './policy'

// Prompt
export { definePrompt } from './prompt'
export type { PromptTemplate } from './prompt'

// Re-export model types for convenience
export type { ModelProfile } from '@magnitudedev/ai'

// Slot resolution
export { resolveSlotModel } from './slot-resolution'
export type { UserSlotConfig, ResolvedSlotModel } from './slot-resolution'

// Roles
export {
  createRoles,
  createLeaderRole,
  createScoutRole,
  createArchitectRole,
  createEngineerRole,
  createCriticRole,
  createScientistRole,
  createArtisanRole,
  createAdvisorRole,
} from './roles/index'

// Prompts
import advisorPromptRaw from './prompts/advisor.txt' with { type: 'text' }
import observerPromptRaw from './prompts/observer.txt' with { type: 'text' }
import { definePrompt } from './prompt'
export const advisorPrompt = definePrompt<never>(advisorPromptRaw)
export const observerPrompt = definePrompt<never>(observerPromptRaw)
