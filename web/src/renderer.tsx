/**
 * Browser renderer entry — spec §4.3
 *
 * The dev server owns the browser's same-origin boundary: `/acn/ensure`
 * selects or launches the daemon, while `/rpc`, `/health`, and `/logs` proxy
 * ACN traffic. The browser never connects cross-origin to the ACN.
 */
import { createRoot } from "react-dom/client"
import { RegistryProvider } from "@effect-atom/atom-react"
import { Effect, Layer } from "effect"
import { FetchHttpClient } from "@effect/platform"
import {
  App,
  PlatformProvider,
  createAgentClient,
  AgentClientProvider,
  AcnStartupProvider,
  initializeAppearance,
  createBrowserPlatform,
  createBrowserAcnConnection,
  stopDisplayViewController,
} from "@magnitudedev/web"
import "./styles/tailwind.css"

initializeAppearance()

const root = createRoot(document.getElementById("root")!)

async function main() {
  const platform = createBrowserPlatform()
  const connection = await createBrowserAcnConnection("")
  const initialAcnLifecycle = await Effect.runPromise(
    connection.startup.prepare
  )
  const agentClientTag = createAgentClient(connection.protocolLayer.pipe(
    Layer.provide(FetchHttpClient.layer),
  ))

  root.render(
    <PlatformProvider platform={platform}>
      <AcnStartupProvider startup={connection.startup}>
        <RegistryProvider defaultIdleTTL={5000}>
          <AgentClientProvider tag={agentClientTag}>
            <App initialAcnLifecycle={initialAcnLifecycle} />
          </AgentClientProvider>
        </RegistryProvider>
      </AcnStartupProvider>
    </PlatformProvider>
  )
}

// Clean up stream fiber on page close / reload
window.addEventListener("beforeunload", () => {
  stopDisplayViewController()
})

main()
