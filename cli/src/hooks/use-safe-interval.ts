import { useEffect, useMemo, useRef } from 'react'
import { useMountedRef } from './use-mounted-ref'

export interface SafeIntervalApi {
  set(callback: () => void, delayMs: number): ReturnType<typeof setInterval> | null
  clear(handle: ReturnType<typeof setInterval> | null | undefined): void
  clearAll(): void
}

export function useSafeInterval(): SafeIntervalApi {
  const mountedRef = useMountedRef()
  const handlesRef = useRef(new Set<ReturnType<typeof setInterval>>())

  const api = useMemo(() => {
    const clear = (handle: ReturnType<typeof setInterval> | null | undefined): void => {
      if (handle == null) return
      clearInterval(handle)
      handlesRef.current.delete(handle)
    }

    const clearAll = (): void => {
      for (const handle of handlesRef.current) {
        clearInterval(handle)
      }
      handlesRef.current.clear()
    }

    return {
      set: (callback: () => void, delayMs: number): ReturnType<typeof setInterval> | null => {
        if (!mountedRef.current) return null

        const handle = setInterval(() => {
          if (!mountedRef.current) return
          callback()
        }, delayMs)

        handlesRef.current.add(handle)
        return handle
      },
      clear,
      clearAll,
    }
  }, [mountedRef])

  useEffect(() => {
    return () => {
      api.clearAll()
    }
  }, [api])

  return api
}