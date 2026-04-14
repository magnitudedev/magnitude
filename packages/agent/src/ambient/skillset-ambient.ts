import { Ambient, AmbientServiceTag } from '@magnitudedev/event-core'
import { Context, Effect } from 'effect'
import type { Skillset } from '@magnitudedev/skills'
import { EmptySkillset, SkillsetResolver } from '@magnitudedev/skills'
import { logger } from '@magnitudedev/logger'

export class SelectedSkillsetName extends Context.Tag('SelectedSkillsetName')<
  SelectedSkillsetName,
  string | null
>() {}

export const SkillsetAmbient = Ambient.define<Skillset, SkillsetResolver | SelectedSkillsetName>({
  name: 'Skillset',
  initial: Effect.gen(function* () {
    const name = yield* SelectedSkillsetName
    if (!name) return EmptySkillset
    const resolver = yield* SkillsetResolver
    return yield* resolver.resolve(name).pipe(
      Effect.catchAll((err) => Effect.gen(function* () {
        logger.warn({ err, skillset: name }, 'Failed to resolve skillset, running without one')
        return EmptySkillset
      }))
    )
  }),
})

export function publishSkillset(skillset: Skillset) {
  return Effect.gen(function* () {
    const ambientService = yield* AmbientServiceTag
    yield* ambientService.update(SkillsetAmbient, skillset)
  })
}
