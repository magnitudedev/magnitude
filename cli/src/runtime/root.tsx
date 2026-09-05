import type { ReactNode } from "react"
import type {
  AgentClient,
  Platform,
} from "@magnitudedev/client-common"
import type { ServiceStartup } from "@magnitudedev/client-common"
import {
  ServiceStartupProvider,
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
  readonly startup: ServiceStartup
  readonly app: CliAppProps
}): ReactNode {
  return (
    <PlatformProvider platform={platform}>
      <ServiceStartupProvider startup={startup}>
        <AgentClientProvider tag={agentClient}>
          <DisplayViewControllerProvider>
            <CliApp {...app} />
          </DisplayViewControllerProvider>
        </AgentClientProvider>
      </ServiceStartupProvider>
    </PlatformProvider>
  )
}
