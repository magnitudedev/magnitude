/**
 * ACN client context. Domains use either AtomRpc or Effect Query according to
 * their state ownership; a single domain never uses both.
 *
 * At renderer startup:
 * 1. Call createAgentClient(protocolLayer)
 * 2. Wrap the app in <AgentClientProvider tag={tag}>
 * 3. Effect Query domains materialize static definitions through .effectQuery
 */
import { createContext, useContext, type ReactNode } from "react"
import type { AgentClientInstance } from "./agent-client"

const AgentClientContext = createContext<AgentClientInstance | null>(null)

export interface AgentClientProviderProps {
  readonly tag: AgentClientInstance
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
export function useAgentClient(): AgentClientInstance {
  const client = useContext(AgentClientContext)
  if (!client) {
    throw new Error("useAgentClient must be used within an AgentClientProvider")
  }
  return client
}
