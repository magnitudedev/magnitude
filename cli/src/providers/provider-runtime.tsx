import { createContext, useContext, type ReactNode } from 'react'
import type { ProviderClient } from '@magnitudedev/providers'
import type { MagnitudeSlot } from '@magnitudedev/agent'

const ProviderRuntimeContext = createContext<ProviderClient<MagnitudeSlot> | null>(null)

export function ProviderRuntimeProvider({
  runtime,
  children,
}: {
  runtime: ProviderClient<MagnitudeSlot>
  children: ReactNode
}) {
  return (
    <ProviderRuntimeContext.Provider value={runtime}>
      {children}
    </ProviderRuntimeContext.Provider>
  )
}

export function useProviderRuntime(): ProviderClient<MagnitudeSlot> {
  const runtime = useContext(ProviderRuntimeContext)
  if (!runtime) {
    throw new Error('ProviderRuntimeProvider is missing')
  }
  return runtime
}