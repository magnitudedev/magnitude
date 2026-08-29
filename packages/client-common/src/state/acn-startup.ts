import { createContext, createElement, useContext, type ReactNode } from "react"
import type { AcnStartup } from "@magnitudedev/sdk"

const AcnStartupContext = createContext<AcnStartup | null>(null)

export function AcnStartupProvider({
  startup,
  children,
}: {
  readonly startup: AcnStartup
  readonly children: ReactNode
}): ReactNode {
  return createElement(AcnStartupContext.Provider, { value: startup }, children)
}

export function useAcnStartup(): AcnStartup {
  const startup = useContext(AcnStartupContext)
  if (startup === null) {
    throw new Error("useAcnStartup must be used within an AcnStartupProvider")
  }
  return startup
}
