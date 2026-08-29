import type { ReactNode } from "react"
import type {
  AgentClient,
  Platform,
} from "@magnitudedev/client-common"
import type { AcnStartup } from "@magnitudedev/sdk"
import {
  AcnStartupProvider,
  AgentClientProvider,
  DisplayViewControllerProvider,
  PlatformProvider,
} from "@magnitudedev/client-common"
import { CliApp, type CliAppProps } from "../app"

export function CliApplicationRoot({
  platform,
  agentClient,
  startup,
  app,
}: {
  readonly platform: Platform
  readonly agentClient: AgentClient
  readonly startup: AcnStartup
  readonly app: CliAppProps
}): ReactNode {
  return (
    <PlatformProvider platform={platform}>
      <AcnStartupProvider startup={startup}>
        <AgentClientProvider tag={agentClient}>
          <DisplayViewControllerProvider>
            <CliApp {...app} />
          </DisplayViewControllerProvider>
        </AgentClientProvider>
      </AcnStartupProvider>
    </PlatformProvider>
  )
}
