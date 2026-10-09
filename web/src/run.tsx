import "./styles/tailwind.css"
import { createRoot } from "react-dom/client"
import { useMemo } from "react"
import { RegistryProvider, Result, useAtomValue } from "@effect-atom/atom-react"
import { Deferred, Effect, Either, Exit, Layer, Option, Runtime, Scope } from "effect"
import { FetchHttpClient } from "@effect/platform"
import { MagnitudeClient, readRemoteAccess } from "@magnitudedev/sdk"
import {
  AgentClientProvider, ApplicationSession, createAgentClient, makeFirstPartyConnection, useAgentClient,
  type ApplicationHost,
} from "@magnitudedev/client-common"
import { initializeAppearance } from "./stores/appearance-store"
import { appearanceReadError } from "./appearance"
import { SessionProvider } from "./session"
import { ServiceConnectionProvider } from "./service-view"
import { App, AppShell } from "./app"
import { ModelsSkeleton } from "./components/page-skeletons"
import { ErrorNotice } from "./components/error-notice"
import { RemoteAccessProvider, SignIn } from "./remote-access"

export interface ApplicationEntry<E> {
  readonly host: ApplicationHost
  /** The Magnitude service this window manages. */
  readonly origin: Effect.Effect<string, E>
  readonly navigation: "memory" | "location"
  /** Whether this window may be on another device and so may need to sign in; a browser may. */
  readonly canSignIn: boolean
  /** A desktop window reconnects whenever its owner restarts the service; a browser retries on its own schedule. */
  readonly reconnect: "Automatic" | "OnConnect"
}

function SessionGate() {
  const client = useAgentClient()
  const session = useAtomValue(useMemo(() => client.runtime.atom(ApplicationSession), [client]))
  if (Result.isSuccess(session)) return <SessionProvider session={session.value}><App /></SessionProvider>
  if (Result.isFailure(session)) return <div className="p-6"><ErrorNotice title="Magnitude couldn’t open" description="Reload this window. If it still won’t open, restart Magnitude." /></div>
  return null
}

/** Renders the app into `#root` for one host; this is the entry's single Effect boundary. */
export const renderApplication = <E,>(entry: ApplicationEntry<E>) => {
  const root = createRoot(document.getElementById("root")!)
  const platform = Option.getOrUndefined(Option.map(entry.host.window, window => window.platform))
  if (platform !== undefined) document.documentElement.dataset.desktopPlatform = platform
  const boot = Effect.gen(function* () {
    const appearance = yield* entry.host.appearance.read.pipe(Effect.either)
    initializeAppearance(Either.isRight(appearance) ? appearance.right : "system")
    root.render(<AppShell page="discover" platform={platform}><ModelsSkeleton page="discover" /></AppShell>)
    const origin = yield* entry.origin
    // A browser on another device signs in before it connects. If the service can't be reached yet,
    // the app shows that, and asks for the key once the service answers again.
    const status = entry.canSignIn ? yield* readRemoteAccess(origin).pipe(Effect.provide(FetchHttpClient.layer), Effect.option) : Option.none()
    if (Option.isSome(status) && status.value._tag === "SignInRequired") {
      const signedIn = yield* Deferred.make<void>()
      const runtime = yield* Effect.runtime<never>()
      root.render(<SignIn origin={origin} keyConfigured={status.value.keyConfigured} onSignedIn={() => { Runtime.runFork(runtime)(Deferred.succeed(signedIn, undefined)) }} />)
      yield* Deferred.await(signedIn)
    }
    const remote = Option.isSome(status) && status.value._tag !== "Local"
    const scope = yield* Scope.make()
    const runtime = yield* Effect.runtime<never>()
    window.addEventListener("beforeunload", () => { Runtime.runFork(runtime)(Scope.close(scope, Exit.void)) }, { once: true })
    const connection = yield* makeFirstPartyConnection(MagnitudeClient.layer({ origin, autoStart: false, reconnect: entry.reconnect }).pipe(Layer.provide(FetchHttpClient.layer))).pipe(Effect.provideService(Scope.Scope, scope))
    const client = createAgentClient(connection.client, { host: entry.host, navigation: entry.navigation })
    const readError = Either.isLeft(appearance) ? "The saved appearance could not be read. Using System appearance." : null
    root.render(<RegistryProvider initialValues={[[appearanceReadError, readError]]}><AgentClientProvider tag={client}><ServiceConnectionProvider connection={connection.client.connection}><RemoteAccessProvider access={{ origin, canSignIn: entry.canSignIn, remote }}><SessionGate /></RemoteAccessProvider></ServiceConnectionProvider></AgentClientProvider></RegistryProvider>)
  })
  Effect.runPromise(boot).catch(error => {
    console.error(error)
    root.render(<div className="p-6"><ErrorNotice title="Magnitude couldn’t open" description="Quit Magnitude and open it again." /></div>)
  })
}
