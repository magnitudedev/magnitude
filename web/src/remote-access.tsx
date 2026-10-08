import { createContext, useContext, useMemo, useState, type ReactNode } from "react"
import { Atom, Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { Effect, Option } from "effect"
import { FetchHttpClient, type HttpClient } from "@effect/platform"
import { readRemoteAccess, signInRemotely, signOutRemotely, type InvalidServiceResponse, type ServiceUnavailable, type SignInFailure } from "@magnitudedev/sdk"
import { MagnitudeMark } from "./components/magnitude-mark"
import { ErrorNotice } from "./components/error-notice"
import { ConfirmDialog } from "./components/confirm-dialog"
import { Button } from "./components/ui/button"
import { Input } from "./components/ui/input"

/**
 * How this window reaches its service. `canSignIn` is true for a browser, which may be on another
 * device; `remote` is true once that browser is signed in from another device.
 */
export interface RemoteAccess {
  readonly origin: string
  readonly canSignIn: boolean
  readonly remote: boolean
}

const RemoteAccessContext = createContext<RemoteAccess>({ origin: "", canSignIn: false, remote: false })

export function RemoteAccessProvider({ access, children }: { access: RemoteAccess; children: ReactNode }) {
  return <RemoteAccessContext.Provider value={access}>{children}</RemoteAccessContext.Provider>
}

export const useRemoteAccess = (): RemoteAccess => useContext(RemoteAccessContext)

const withFetch = <A, E>(effect: Effect.Effect<A, E, HttpClient.HttpClient>) => effect.pipe(Effect.provide(FetchHttpClient.layer))

/** Whether the service now asks this browser to sign in, as it does after it restarts. */
export const signInRequired = (origin: string) => withFetch(readRemoteAccess(origin)).pipe(
  Effect.map(status => status._tag === "SignInRequired"),
  Effect.orElseSucceed(() => false),
)

/** Reloading starts the window again, which asks for the key when the service requires it. */
export const restartWindow = Effect.sync(() => window.location.reload())

export const signOut = (origin: string) => withFetch(signOutRemotely(origin)).pipe(Effect.zipRight(restartWindow))

type SignInError = SignInFailure | ServiceUnavailable | InvalidServiceResponse

const failureMessage = (failure: Option.Option<SignInError>) => {
  const value = Option.getOrUndefined(failure)
  if (value?._tag === "SignInThrottled") return { title: "Too many attempts", description: `Wait ${value.retryAfterSeconds === 1 ? "a second" : `${value.retryAfterSeconds} seconds`}, then try again.` }
  if (value?._tag !== "SignInRefused") return { title: "Can’t reach Magnitude", description: "Check that Magnitude is running on that computer and that this device can reach it." }
  switch (value.reason) {
    case "WrongKey": return { title: "That key didn’t work", description: "Copy the key again from Settings → Network access on the computer running Magnitude." }
    case "NoKey": return { title: "No key is set", description: "On the computer running Magnitude, open Settings → Network access and choose Regenerate to create one." }
    case "NetworkAccessOff": return { title: "Network access is off", description: "Turn on Network access in Settings on the computer running Magnitude." }
    case "CrossOrigin": return { title: "Couldn’t sign in", description: "Open Magnitude using its own address, then try again." }
  }
}

/** The page a browser on another device shows until it signs in with the Network access key. */
export function SignIn({ origin, keyConfigured, onSignedIn }: { origin: string; keyConfigured: boolean; onSignedIn: () => void }) {
  const [key, setKey] = useState("")
  const action = useMemo(() => Atom.fn((value: string) => withFetch(signInRemotely(origin, value)).pipe(Effect.tap(() => Effect.sync(onSignedIn)))), [origin, onSignedIn])
  const submit = useAtomSet(action)
  const result = useAtomValue(action)
  const error = Result.isFailure(result) && !result.waiting ? failureMessage(Result.error(result)) : undefined
  return <main className="flex min-h-dvh items-center justify-center bg-slate-50 px-4 py-10 text-slate-900 dark:bg-slate-925 dark:text-slate-100">
    <div className="w-full max-w-sm">
      <div className="mb-6 flex items-center gap-2.5"><MagnitudeMark className="size-8" /><span className="font-heading text-lg font-semibold">Magnitude</span></div>
      <form className="rounded-lg border border-slate-300 bg-white p-6 dark:border-slate-750 dark:bg-slate-850" onSubmit={event => { event.preventDefault(); if (key.trim()) submit(key.trim()) }}>
        <h1 className="font-heading text-xl font-semibold tracking-tight">Sign in</h1>
        {keyConfigured ? <>
          <p className="mt-2 text-sm text-slate-500">Enter the Network access key from Settings on the computer running Magnitude.</p>
          <label className="mt-5 block text-sm font-medium" htmlFor="network-access-key">Network access key</label>
          <Input id="network-access-key" type="password" autoComplete="current-password" autoFocus spellCheck={false} value={key} onChange={event => setKey(event.target.value)} className="mt-1.5 font-mono" />
          {error && <ErrorNotice className="mt-4" title={error.title} description={error.description} />}
          <Button type="submit" className="mt-5 w-full" disabled={!key.trim() || result.waiting}>Sign in</Button>
        </> : <ErrorNotice className="mt-4" title="No key is set" description="On the computer running Magnitude, open Settings → Network access and choose Regenerate to create one, then reload this page." />}
      </form>
    </div>
  </main>
}

/** When the browser loses its connection, find out whether the service now needs a sign-in. */
export function useSignInRecheck(): Effect.Effect<boolean> {
  const access = useRemoteAccess()
  return useMemo(() => access.canSignIn
    ? signInRequired(access.origin).pipe(Effect.tap(required => required ? restartWindow : Effect.void))
    : Effect.succeed(false), [access])
}

export type DisconnectWarning = "TurnOffNetworkAccess" | "RegenerateKey" | "ChangeAddress" | "Restart"

const warnings: Record<DisconnectWarning, { title: string; description: string; confirmLabel: string }> = {
  TurnOffNetworkAccess: {
    title: "Turn off network access?",
    description: "This browser is connected over the network. When Magnitude restarts, it stops accepting other devices, and this browser will be disconnected.",
    confirmLabel: "Turn off",
  },
  RegenerateKey: {
    title: "Regenerate the key?",
    description: "When Magnitude restarts, the current key stops working. This browser will be signed out, and you’ll need the new key to sign in again.",
    confirmLabel: "Regenerate",
  },
  ChangeAddress: {
    title: "Change the address?",
    description: "When Magnitude restarts, it accepts connections only on the address you choose. This browser may not be able to reconnect.",
    confirmLabel: "Change address",
  },
  Restart: {
    title: "Restart Magnitude?",
    description: "This browser will be disconnected while Magnitude restarts, and you’ll need to sign in again.",
    confirmLabel: "Restart",
  },
}

/**
 * Runs an action that can cut this browser off. From another device it first asks in an
 * AlertDialog; on the server's own machine it runs at once.
 */
export function useDisconnectWarning(): { readonly warn: (warning: DisconnectWarning, action: () => void) => void; readonly dialog: ReactNode } {
  const { remote } = useRemoteAccess()
  const [pending, setPending] = useState<{ warning: DisconnectWarning; action: () => void } | null>(null)
  const copy = pending === null ? null : warnings[pending.warning]
  return {
    warn: (warning, action) => remote ? setPending({ warning, action }) : action(),
    dialog: copy && <ConfirmDialog open onOpenChange={open => { if (!open) setPending(null) }} title={copy.title} description={copy.description}
      confirmLabel={copy.confirmLabel} onConfirm={() => { const action = pending?.action; setPending(null); action?.() }} />,
  }
}

/** The viewer's operating system, for commands they run on this device rather than the server. */
export const viewerPlatform = (): string => /Windows/i.test(navigator.userAgent) ? "win32" : /Mac/i.test(navigator.userAgent) ? "darwin" : "linux"
