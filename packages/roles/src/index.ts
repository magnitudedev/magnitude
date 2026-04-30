// Types
export type { Slot, PolicyContext, PolicyRule, RoleDefinition } from './types'
export { SLOTS, isSlot } from './types'

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

// Model resolution
export type { ModelOverrideEntry, ModelOverrides } from './model-resolver'
export { resolveModel } from './model-resolver'

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
