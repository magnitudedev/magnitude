import { createContext, useContext, type ReactNode } from 'react'
import type { ProviderClient } from '@magnitudedev/providers'

const ProviderRuntimeContext = createContext<ProviderClient | null>(null)

export function ProviderRuntimeProvider({
  runtime,
  children,
}: {
  runtime: ProviderClient
  children: ReactNode
}) {
  return (
    <ProviderRuntimeContext.Provider value={runtime}>
      {children}
    </ProviderRuntimeContext.Provider>
  )
}

export function useProviderRuntime(): ProviderClient {
  const runtime = useContext(ProviderRuntimeContext)
  if (!runtime) {
    throw new Error('ProviderRuntimeProvider is missing')
  }
  return runtime
}