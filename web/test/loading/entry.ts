import { createElement } from "react"
import { createRoot } from "react-dom/client"
import { RegistryContext } from "@effect-atom/atom-react"
import "../../src/styles/tailwind.css"
import { initializeAppearance } from "../../src/stores/appearance-store"
import { SessionProvider } from "../../src/session"
import { ServiceConnectionProvider } from "../../src/service-view"
import { Effect, Stream } from "effect"
import { App } from "../../src/app"
import { registry, service } from "./client"

// Renders the real App in the fixture registry and session, without starting the service.
const browserEntry = new URLSearchParams(location.search).get("entry") === "browser"
if (!browserEntry) document.documentElement.dataset.desktopPlatform = "darwin"
const connection = { state: Effect.never, changes: Stream.make({ _tag: "Ready" } as any), connect: Effect.void } as any
initializeAppearance(new URLSearchParams(location.search).get("theme") === "dark" ? "dark" : "light")
createRoot(document.getElementById("root")!).render(
  createElement(RegistryContext.Provider, { value: registry }, createElement(ServiceConnectionProvider, { connection, children: createElement(SessionProvider, { session: service }, createElement(App)) })),
)
