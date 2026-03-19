import { init } from '@paralleldrive/cuid2'

export const createId = init({ length: 12 })
export const createShortId = init({ length: 8 })

/**
 * Generates a lexicographically sortable ID based on the current timestamp.
 * Uses base36-encoded epoch milliseconds, producing ~8 character alphanumeric strings.
 * Alphabetical sort order equals chronological order.
 */
export function generateSortableId(): string {
  return Date.now().toString(36)
}