import { useMemo } from 'react'
import { useMountedRef } from './use-mounted-ref'

export interface UnmountSignal {
  readonly mountedRef: React.RefObject<boolean>
  isMounted(): boolean
  guard<T>(fn: () => T): T | undefined
}

export function useUnmountSignal(): UnmountSignal {
  const mountedRef = useMountedRef()

  return useMemo(() => ({
    mountedRef,
    isMounted: () => mountedRef.current,
    guard: <T>(fn: () => T): T | undefined => {
      if (!mountedRef.current) return undefined
      return fn()
    },
  }), [mountedRef])
}