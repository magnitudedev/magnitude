/**
 * Browser renderer entry — spec §4.3
 *
 * The dev server owns the browser's same-origin boundary: `/acn/ensure`
 * selects or launches the daemon, while `/rpc`, `/health`, and `/logs` proxy
 * ACN traffic. The browser never connects cross-origin to the ACN.
 */
import { createRoot } from "react-dom/client"
import { RegistryProvider } from "@effect-atom/atom-react"
import {
  App,
  PlatformProvider,
  createAgentClient,
  AgentClientProvider,
  initializeAppearance,
  createBrowserPlatform,
  stopDisplayViewController,
} from "@magnitudedev/web"
import "./styles/tailwind.css"

initializeAppearance()

const root = createRoot(document.getElementById("root")!)

async function main() {
  const platform = await createBrowserPlatform("")
  const agentClientTag = createAgentClient(platform.protocolLayer)

  root.render(
    <PlatformProvider platform={platform}>
      <RegistryProvider defaultIdleTTL={5000}>
        <AgentClientProvider tag={agentClientTag}>
          <App />
        </AgentClientProvider>
      </RegistryProvider>
    </PlatformProvider>
  )
}

// Clean up stream fiber on page close / reload
window.addEventListener("beforeunload", () => {
  stopDisplayViewController()
})

main()
