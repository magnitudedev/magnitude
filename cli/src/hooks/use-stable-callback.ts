import { useRef } from 'react'

/**
 * Stable-identity callback that always invokes the latest closure. Ref-based,
 * no useEffect — the classic useEvent pattern for handlers passed to
 * imperative subscribers (OpenTUI keyboard/mouse) that must not resubscribe
 * per render.
 */
export function useStableCallback<A extends unknown[], R>(fn: (...args: A) => R): (...args: A) => R {
  const latestRef = useRef(fn)
  latestRef.current = fn
  const stableRef = useRef<((...args: A) => R) | null>(null)
  if (!stableRef.current) {
    stableRef.current = (...args: A) => latestRef.current(...args)
  }
  return stableRef.current
}
