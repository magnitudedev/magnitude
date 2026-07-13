import { createContext, useCallback, useContext, useRef, useSyncExternalStore } from 'react'
import type { DisplayReader, DisplaySpeculator } from './display-view-store'
import type { DisplayViewSnapshot } from '@magnitudedev/sdk'

export const DisplayReaderContext = createContext<DisplayReader | null>(null)
export const DisplaySpeculatorContext = createContext<DisplaySpeculator | null>(null)

export function useDisplayReader(): DisplayReader {
  const reader = useContext(DisplayReaderContext)
  if (!reader) throw new Error('useDisplayReader must be used within DisplayReaderContext.Provider')
  return reader
}

export function useDisplaySpeculator(): DisplaySpeculator {
  const speculator = useContext(DisplaySpeculatorContext)
  if (!speculator) throw new Error('useDisplaySpeculator must be used within DisplaySpeculatorContext.Provider')
  return speculator
}

export function useDisplayView<T>(
  selector: (view: DisplayViewSnapshot) => T,
  isEqual: (left: T, right: T) => boolean = Object.is,
): T {
  const store = useDisplayReader()

  const cacheRef = useRef<{
    readonly source: DisplayViewSnapshot
    readonly selector: (view: DisplayViewSnapshot) => T
    readonly selected: T
  } | null>(null)

  const getSnapshot = useCallback(() => {
    const source = store.getSnapshot()
    const cached = cacheRef.current
    if (cached && cached.source === source && cached.selector === selector) return cached.selected

    const selected = selector(source)
    if (cached && isEqual(cached.selected, selected)) {
      cacheRef.current = { source, selector, selected: cached.selected }
      return cached.selected
    }

    cacheRef.current = { source, selector, selected }
    return selected
  }, [isEqual, selector, store])

  return useSyncExternalStore(store.subscribe, getSnapshot, getSnapshot)
}
