import { createContext, createElement, useContext, type ReactNode } from "react"
import type { ServiceStartup } from "../connection/connection"

const ServiceStartupContext = createContext<ServiceStartup | null>(null)

export function ServiceStartupProvider({
  startup,
  children,
}: {
  readonly startup: ServiceStartup
  readonly children: ReactNode
}): ReactNode {
  return createElement(ServiceStartupContext.Provider, { value: startup }, children)
}

export function useServiceStartup(): ServiceStartup {
  const startup = useContext(ServiceStartupContext)
  if (startup === null) {
    throw new Error("useServiceStartup must be used within an ServiceStartupProvider")
  }
  return startup
}
