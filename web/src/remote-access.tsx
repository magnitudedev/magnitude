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

import { readLanguage, translate } from "./i18n"

const tr = (key: Parameters<typeof translate>[1]) => translate(readLanguage(), key)

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
  if (value?._tag === "SignInThrottled") return { title: tr("tooManyAttempts"), description: tr("waitThenRetry").replace("{duration}", value.retryAfterSeconds === 1 ? tr("aSecond") : `${value.retryAfterSeconds} ${tr("secondsWord")}`) }
  if (value?._tag !== "SignInRefused") return { title: tr("cantReachMagnitude"), description: tr("canNotReachCheck") }
  switch (value.reason) {
    case "WrongKey": return { title: tr("wrongKey"), description: tr("copyKeyAgain") }
    case "NoKey": return { title: tr("noKeySet"), description: tr("noKeySetDescription") }
    case "NetworkAccessOff": return { title: tr("networkAccessOff"), description: tr("turnOnNetworkAccess") }
    case "CrossOrigin": return { title: tr("couldNotSignIn"), description: tr("openOwnAddress") }
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
        <h1 className="font-heading text-xl font-semibold tracking-tight">{tr("signIn")}</h1>
        {keyConfigured ? <>
          <p className="mt-2 text-sm text-slate-500">{tr("enterNetworkKey")}</p>
          <label className="mt-5 block text-sm font-medium" htmlFor="network-access-key">{tr("networkAccessKey")}</label>
          <Input id="network-access-key" type="password" autoComplete="current-password" autoFocus spellCheck={false} value={key} onChange={event => setKey(event.target.value)} className="mt-1.5 font-mono" />
          {error && <ErrorNotice className="mt-4" title={error.title} description={error.description} />}
          <Button type="submit" className="mt-5 w-full" disabled={!key.trim() || result.waiting}>{tr("signIn")}</Button>
        </> : <ErrorNotice className="mt-4" title={tr("noKeySet")} description={`${tr("noKeySetDescription")} ${tr("reloadPage")}`} />}
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

export type DisconnectWarning = "TurnOffNetworkAccess" | "RegenerateKey" | "StopRequiringKey" | "ChangeAddress" | "Restart"

const warnings: Record<DisconnectWarning, { title: string; description: string; confirmLabel: string }> = {
  TurnOffNetworkAccess: { title: tr("turnOffNetworkQuestion"), description: tr("turnOffNetworkDescription"), confirmLabel: tr("turnOff") },
  RegenerateKey: { title: tr("regenerateKeyQuestion"), description: tr("regenerateKeyDescription"), confirmLabel: tr("regenerate") },
  StopRequiringKey: { title: tr("stopRequiringKeyQuestion"), description: tr("stopRequiringKeyDescription"), confirmLabel: tr("stopRequiring") },
  ChangeAddress: { title: tr("changeAddressQuestion"), description: tr("changeAddressDescription"), confirmLabel: tr("changeAddress") },
  Restart: { title: tr("restartMagnitudeQuestion"), description: tr("restartDisconnectDescription"), confirmLabel: tr("restart") },
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
export const viewerPlatform = (): "win32" | "darwin" | "linux" => /Windows/i.test(navigator.userAgent) ? "win32" : /Mac/i.test(navigator.userAgent) ? "darwin" : "linux"
