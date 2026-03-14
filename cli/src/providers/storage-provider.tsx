import { createContext, useContext, type ReactNode } from 'react'
import type { StorageClient } from '@magnitudedev/storage'

const StorageContext = createContext<StorageClient | null>(null)

export function StorageProvider({
  client,
  children,
}: {
  client: StorageClient
  children: ReactNode
}) {
  return (
    <StorageContext.Provider value={client}>
      {children}
    </StorageContext.Provider>
  )
}

export function useStorage(): StorageClient {
  const client = useContext(StorageContext)
  if (!client) {
    throw new Error('useStorage must be used within StorageProvider')
  }
  return client
}