import { Ambient, AmbientServiceTag } from '@magnitudedev/event-core'
import { Effect } from 'effect'
import { defineToolkit, type Toolkit } from '@magnitudedev/harness'

export const ToolkitAmbient = Ambient.define<Toolkit, never>({
  name: 'Toolkit',
  initial: Effect.succeed(defineToolkit({})),
})

export function publishToolkit(toolkit: Toolkit) {
  return Effect.gen(function* () {
    const ambientService = yield* AmbientServiceTag
    yield* ambientService.update(ToolkitAmbient, toolkit)
  })
}
