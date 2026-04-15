
import { Ambient, AmbientServiceTag } from '@magnitudedev/event-core'
import { Effect } from 'effect'
import type { Skill } from '@magnitudedev/skills'

export const SkillsAmbient = Ambient.define<Map<string, Skill>, never>({
  name: 'Skills',
  initial: new Map(),
})

export function publishSkills(skills: Map<string, Skill>) {
  return Effect.gen(function* () {
    const ambientService = yield* AmbientServiceTag
    yield* ambientService.update(SkillsAmbient, skills)
  })
}
