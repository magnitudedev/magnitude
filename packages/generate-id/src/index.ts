import { init } from '@paralleldrive/cuid2'

export const createId = init({ length: 12 })
export const createShortId = init({ length: 8 })