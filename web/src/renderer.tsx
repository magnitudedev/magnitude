/**
 * Browser renderer entry — spec §4.3
 *
 * The dev server (one process, one port) serves the web app AND handles
 * daemon lifecycle. The browser uses relative URLs for everything:
 * - /discover, /spawn → daemon lifecycle (proxied to spawner)
 * - /rpc, /health, /logs → daemon RPC (proxied to daemon)
 *
 * The renderer provides a remote DaemonSpawner backed by the dev server.
 * SDK recovery drives /discover and /spawn through that spawner.
 */
import { createRoot } from "react-dom/client"
import { RegistryProvider } from "@effect-atom/atom-react"
import {
  App,
  PlatformProvider,
  createAgentClient,
  AgentClientProvider,
  injectCssVars,
  createBrowserPlatform,
  stopDisplayViewController,
} from "@magnitudedev/web"
import "./styles/vars.css"
import "./styles/globals.css"

injectCssVars()

const root = createRoot(document.getElementById("root")!)

async function main() {
  const platform = createBrowserPlatform("")
  const agentClientTag = createAgentClient(platform.protocolLayer)

  root.render(
    <PlatformProvider platform={platform}>
      <RegistryProvider defaultIdleTTL={5000}>
        <AgentClientProvider tag={agentClientTag}>
          <App />
        </AgentClientProvider>
      </RegistryProvider>
    </PlatformProvider>,
  )
}

// Clean up stream fiber on page close / reload
window.addEventListener("beforeunload", () => {
  stopDisplayViewController()
})

main()
