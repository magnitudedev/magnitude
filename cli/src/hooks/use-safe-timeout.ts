import { useEffect, useMemo, useRef } from 'react'
import { useMountedRef } from './use-mounted-ref'

export interface SafeTimeoutApi {
  set(callback: () => void, delayMs: number): ReturnType<typeof setTimeout> | null
  clear(handle: ReturnType<typeof setTimeout> | null | undefined): void
  clearAll(): void
}

export function useSafeTimeout(): SafeTimeoutApi {
  const mountedRef = useMountedRef()
  const handlesRef = useRef(new Set<ReturnType<typeof setTimeout>>())

  const api = useMemo(() => {
    const clear = (handle: ReturnType<typeof setTimeout> | null | undefined): void => {
      if (handle == null) return
      clearTimeout(handle)
      handlesRef.current.delete(handle)
    }

    const clearAll = (): void => {
      for (const handle of handlesRef.current) {
        clearTimeout(handle)
      }
      handlesRef.current.clear()
    }

    return {
      set: (callback: () => void, delayMs: number): ReturnType<typeof setTimeout> | null => {
        if (!mountedRef.current) return null

        let handle: ReturnType<typeof setTimeout>
        handle = setTimeout(() => {
          handlesRef.current.delete(handle)
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