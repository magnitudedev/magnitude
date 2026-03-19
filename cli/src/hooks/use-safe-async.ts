import { useCallback, useEffect, useMemo, useRef } from 'react'
import { useMountedRef } from './use-mounted-ref'

export interface SafeAsyncContext {
  readonly mountedRef: React.RefObject<boolean>
  isMounted(): boolean
  checkpoint(): boolean
}

export interface SafeAsyncApi {
  run<T>(fn: (ctx: SafeAsyncContext) => Promise<T>): Promise<T | undefined>
  wrap<Args extends unknown[], T>(
    fn: (ctx: SafeAsyncContext, ...args: Args) => Promise<T>,
  ): (...args: Args) => Promise<T | undefined>
  invalidate(): void
  readonly generationRef: React.RefObject<number>
}

export function useSafeAsync(): SafeAsyncApi {
  const mountedRef = useMountedRef()
  const generationRef = useRef(0)

  const invalidate = useCallback(() => {
    generationRef.current += 1
  }, [])

  const run = useCallback(async <T>(fn: (ctx: SafeAsyncContext) => Promise<T>): Promise<T | undefined> => {
    if (!mountedRef.current) return undefined

    const generation = generationRef.current
    const ctx: SafeAsyncContext = {
      mountedRef,
      isMounted: () => mountedRef.current,
      checkpoint: () => mountedRef.current && generationRef.current === generation,
    }

    const result = await fn(ctx)
    if (!ctx.checkpoint()) return undefined
    return result
  }, [mountedRef])

  const wrap = useCallback(<Args extends unknown[], T>(
    fn: (ctx: SafeAsyncContext, ...args: Args) => Promise<T>,
  ) => {
    return (...args: Args): Promise<T | undefined> => run((ctx) => fn(ctx, ...args))
  }, [run])

  const api = useMemo(() => ({
    run,
    wrap,
    invalidate,
    generationRef,
  }), [run, wrap, invalidate])

  useEffect(() => {
    return () => {
      invalidate()
    }
  }, [invalidate])

  return api
}