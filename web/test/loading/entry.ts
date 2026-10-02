import { createElement } from "react"
import { createRoot } from "react-dom/client"
import { RegistryContext } from "@effect-atom/atom-react"
import "../../src/styles/tailwind.css"
import { initializeAppearance } from "../../src/stores/appearance-store"
import { SessionProvider } from "../../src/session"
import { App } from "../../src/app"
import { registry, service } from "./client"

// Renders the real App in the fixture registry and session, without starting the service.
document.documentElement.dataset.desktopPlatform = "darwin"
initializeAppearance(new URLSearchParams(location.search).get("theme") === "dark" ? "dark" : "light")
createRoot(document.getElementById("root")!).render(
  createElement(RegistryContext.Provider, { value: registry }, createElement(SessionProvider, { session: service }, createElement(App))),
)
