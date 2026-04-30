import type { RoleDefinition, Slot } from '../types'
import { createLeaderRole } from './leader'
import { createScoutRole } from './scout'
import { createArchitectRole } from './architect'
import { createEngineerRole } from './engineer'
import { createCriticRole } from './critic'
import { createScientistRole } from './scientist'
import { createArtisanRole } from './artisan'
import { createAdvisorRole } from './advisor'

export function createRoles(): Record<Slot, RoleDefinition> {
  return {
    leader: createLeaderRole(),
    scout: createScoutRole(),
    architect: createArchitectRole(),
    engineer: createEngineerRole(),
    critic: createCriticRole(),
    scientist: createScientistRole(),
    artisan: createArtisanRole(),
    advisor: createAdvisorRole(),
  }
}

export {
  createLeaderRole,
  createScoutRole,
  createArchitectRole,
  createEngineerRole,
  createCriticRole,
  createScientistRole,
  createArtisanRole,
  createAdvisorRole,
}
