import { Ambient, AmbientServiceTag } from '@magnitudedev/event-core'
import { Effect } from 'effect'
import type { Skillset } from '@magnitudedev/skills'
import { SkillsetResolver } from '@magnitudedev/skills'

export const SkillsetAmbient = Ambient.define<Skillset, SkillsetResolver>({
  name: 'Skillset',
  initial: Effect.gen(function* () {
    const resolver = yield* SkillsetResolver
    return yield* resolver.resolve('magnitude')
  }).pipe(Effect.orDie),
})

export function publishSkillset(skillset: Skillset) {
  return Effect.gen(function* () {
    const ambientService = yield* AmbientServiceTag
    yield* ambientService.update(SkillsetAmbient, skillset)
  })
}
