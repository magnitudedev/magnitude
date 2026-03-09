export { createId } from '@magnitudedev/generate-id'

export function tupleMap<T, U>(
  items: readonly [T, ...T[]],
  f: (item: T) => U
): [U, ...U[]] {
  const [head, ...tail] = items
  return [f(head), ...tail.map(f)]
}

export function isNonEmpty<T>(arr: readonly T[]): arr is readonly [T, ...T[]] {
  return arr.length > 0
}