import { describe, expectTypeOf, it } from 'vitest'
import type { TaskUpdated } from '../../../events'
import type { ValidatedTaskGraphEvent } from '../events'

describe('validated task graph ingress typing', () => {
  it('does not allow raw TaskUpdated as ValidatedTaskGraphEvent', () => {
    expectTypeOf<TaskUpdated>().not.toMatchTypeOf<ValidatedTaskGraphEvent>()
  })
})
