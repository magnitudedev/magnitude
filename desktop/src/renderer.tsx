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
import { Effect } from "effect"
import {
  App,
  PlatformProvider,
  createAgentClient,
  AgentClientProvider,
  initializeAppearance,
  MagnitudeMark,
  stopDisplayViewController,
} from "@magnitudedev/web"
import { DaemonConnectionError } from "@magnitudedev/web"
import { createDesktopPlatform } from "./platform"
import "@web-styles/tailwind.css"

initializeAppearance()

const desktopApi = window.__magnitudeDesktop
const root = createRoot(document.getElementById("root")!)
let activePlatform:
  | Awaited<ReturnType<typeof createDesktopPlatform>>
  | undefined

document.documentElement.dataset.desktopPlatform = desktopApi.platform

function renderLoading() {
  root.render(
    <div className="flex h-screen flex-col items-center justify-center bg-slate-50 font-sans text-slate-900 dark:bg-slate-875 dark:text-slate-200">
      <MagnitudeMark className="mb-6 h-auto w-[82px]" />
      <div className="font-heading text-[30px] font-semibold leading-tight tracking-[-0.025em]">
        Opening Magnitude
      </div>
      <div className="mt-5 flex items-center gap-2.5 text-[16px] leading-7 text-slate-600 dark:text-slate-300">
        <div className="size-[17px] rounded-full border-2 border-slate-300 border-t-blue-700 animate-spin motion-reduce:animate-none dark:border-slate-750 dark:border-t-blue-500" />
        <span>Initializing desktop services…</span>
      </div>
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
  const initialAcnLifecycle = await Effect.runPromise(
    platform.acnStartup.prepare
  )
  const agentClientTag = createAgentClient(platform.protocolLayer)
  root.render(
    <PlatformProvider platform={platform}>
      <RegistryProvider defaultIdleTTL={5000}>
        <AgentClientProvider tag={agentClientTag}>
          <App initialAcnLifecycle={initialAcnLifecycle} />
        </AgentClientProvider>
      </RegistryProvider>
    </PlatformProvider>
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
