import { useCallback, useRef } from 'react'
import { useMountedRef } from './use-mounted-ref'

export function useSafeEvent<Args extends unknown[]>(
  fn: (...args: Args) => void,
): (...args: Args) => void {
  const mountedRef = useMountedRef()
  const fnRef = useRef(fn)
  fnRef.current = fn

  return useCallback((...args: Args) => {
    if (!mountedRef.current) return
    fnRef.current(...args)
  }, [mountedRef])
}