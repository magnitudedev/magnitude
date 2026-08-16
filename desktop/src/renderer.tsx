/**
 * Desktop renderer entry — spec §5.2
 *
 * Reads `window.__magnitudeDesktop`, creates the DesktopPlatform,
 * creates the AgentClient AtomRpc tag with the desktop daemon services, and mounts App
 * inside PlatformProvider + RegistryProvider + AgentClientProvider.
 *
 * The scoped client lease is the first RPC consumer and establishes ACN
 * selection as the interactive client lifetime starts.
 *
 * On window close, interrupts the renderer stream and notifies main (§5.6).
 */
import { createRoot } from "react-dom/client"
import { RegistryProvider } from "@effect-atom/atom-react"
import { App, PlatformProvider, createAgentClient, AgentClientProvider, initializeAppearance, stopDisplayViewController } from "@magnitudedev/web"
import { DaemonConnectionError } from "@magnitudedev/web"
import { createDesktopPlatform } from "./platform"
import "@web-styles/vars.css"
import "@web-styles/globals.css"

initializeAppearance()

const desktopApi = window.__magnitudeDesktop
const root = createRoot(document.getElementById("root")!)
let activePlatform: Awaited<ReturnType<typeof createDesktopPlatform>> | undefined

document.documentElement.dataset.desktopPlatform = desktopApi.platform

function renderLoading() {
  root.render(
    <div style={{
      display: "flex",
      height: "100vh",
      alignItems: "center",
      justifyContent: "center",
      background: "var(--bg-base)",
      color: "var(--fg-secondary)",
      fontFamily: "var(--font-sans)",
      fontSize: 14,
    }}>
      Connecting to Magnitude daemon...
    </div>
  )
}

function renderDaemonError(message: string) {
  root.render(
    <DaemonConnectionError
      message={message}
      reconnecting={false}
      onRetry={() => {
        // Retry: reload the app to re-attempt daemon connection
        window.location.reload()
      }}
      onQuit={() => {
        desktopApi.quit()
      }}
    />
  )
}

async function renderApp() {
  const platform = await createDesktopPlatform(desktopApi)
  activePlatform = platform
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

// On window close, interrupt the stream fiber and notify main (§5.6)
window.addEventListener("beforeunload", () => {
  stopDisplayViewController()
  void activePlatform?.shutdown()
  desktopApi.interruptStream()
})

renderLoading()
renderApp().catch((error: Error) => {
  renderDaemonError(error.message)
})
