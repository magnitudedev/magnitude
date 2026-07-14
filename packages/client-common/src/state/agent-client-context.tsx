/**
 * AgentClient runtime — stores the created AtomRpc tag instance and
 * provides it to components via React context.
 *
 * At renderer startup:
 * 1. Call createAgentClient(protocolLayer) — this creates the AtomRpc.Tag and
 *    calls Atom.runtime.addGlobalLayer(tag.layer)
 * 2. Wrap the app in <AgentClientProvider tag={tag}>
 * 3. Components use useAgentClient() to call .query() and .mutation()
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
 * Get the AgentClient AtomRpc tag from context.
 * Use this to call .query() and .mutation() in components.
 */
export function useAgentClient(): AgentClientInstance {
  const tag = useContext(AgentClientContext)
  if (!tag) {
    throw new Error("useAgentClient must be used within an AgentClientProvider")
  }
  return tag
}
