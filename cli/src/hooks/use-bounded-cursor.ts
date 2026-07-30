import { useCallback, useState } from "react"

const clampIndex = (index: number, size: number): number =>
  Math.max(0, Math.min(Math.max(0, size - 1), index))

export interface BoundedCursor {
  readonly index: number
  readonly previous: () => void
  readonly next: () => void
  readonly select: (index: number) => void
  readonly reset: () => void
}

export const useBoundedCursor = (
  size: number,
  initialIndex = 0,
): BoundedCursor => {
  const [storedIndex, setStoredIndex] = useState(initialIndex)
  const index = clampIndex(storedIndex, size)
  const previous = useCallback(() => {
    setStoredIndex((current) => clampIndex(current - 1, size))
  }, [size])
  const next = useCallback(() => {
    setStoredIndex((current) => clampIndex(current + 1, size))
  }, [size])
  const select = useCallback((selected: number) => {
    setStoredIndex(clampIndex(selected, size))
  }, [size])
  const reset = useCallback(() => {
    setStoredIndex(0)
  }, [])

  return { index, previous, next, select, reset }
}
