/**
 * ACN client context: the connection's Effect Query client, provided once at
 * renderer startup.
 *
 * 1. Call createAgentClient(protocolLayer)
 * 2. Wrap the app in <AgentClientProvider tag={client}>
 * 3. Hooks and services materialize boundary operations through the client
 */
import { createContext, useContext, type ReactNode } from "react"
import type { AgentClient } from "./agent-client"

const AgentClientContext = createContext<AgentClient | null>(null)

export interface AgentClientProviderProps {
  readonly tag: AgentClient
  readonly children: ReactNode
}

export function AgentClientProvider({ tag, children }: AgentClientProviderProps): ReactNode {
  return (
    <AgentClientContext.Provider value={tag}>
      {children}
    </AgentClientContext.Provider>
  )
}

/**
 * Get the connection client from context.
 */
export function useAgentClient(): AgentClient {
  const client = useContext(AgentClientContext)
  if (!client) {
    throw new Error("useAgentClient must be used within an AgentClientProvider")
  }
  return client
}
