import { createContext, useContext, useMemo, type ReactNode } from "react"
import { Atom, Result, useAtomValue } from "@effect-atom/atom-react"
import { Effect, Option, Schedule, Stream, SubscriptionRef } from "effect"
import type { MagnitudeConnection } from "@magnitudedev/sdk"
import type { TrayRegistration } from "@magnitudedev/sdk/desktop-host"
import { useSession } from "./session"
import { useSignInRecheck } from "./remote-access"

/**
 * The service as this window can know it. A desktop window asks its owner, which still knows the
 * service when ACN is down; a browser only knows whether it can reach ACN.
 */
export type ServiceView =
  | { readonly _tag: "Ready" }
  | { readonly _tag: "Starting" }
  | { readonly _tag: "Failed"; readonly message: string }
  | { readonly _tag: "CleanupFailed"; readonly message: string }
  | { readonly _tag: "Reconnecting" }
  | { readonly _tag: "Unreachable" }

export interface ServiceObservation {
  readonly service: Result.Result<ServiceView, unknown>
  readonly tray: Option.Option<TrayRegistration>
  /** Whether this window can restart a failed service; only its owner can. */
  readonly canRetry: boolean
}

const ConnectionContext = createContext<MagnitudeConnection | null>(null)

export function ServiceConnectionProvider({ connection, children }: { connection: MagnitudeConnection; children: ReactNode }) {
  return <ConnectionContext.Provider value={connection}>{children}</ConnectionContext.Provider>
}

export const useServiceConnection = (): MagnitudeConnection => {
  const connection = useContext(ConnectionContext)
  if (connection === null) throw new Error("useServiceConnection is used outside ServiceConnectionProvider")
  return connection
}

type Reachability = "Connected" | "Reconnecting" | "Unreachable"

/**
 * A browser loses its connection whenever the service restarts. It retries on its own for 30 seconds,
 * which covers a restart, then gives up until someone presses Reconnect. Each attempt first asks
 * whether the service now wants this browser to sign in, as it does after any restart.
 */
const browserReachability = (connection: MagnitudeConnection, attempt: Effect.Effect<void, unknown>) => Stream.unwrapScoped(Effect.gen(function* () {
  const reachability = yield* SubscriptionRef.make<Reachability>("Connected")
  yield* connection.changes.pipe(Stream.runForEach(state => Effect.gen(function* () {
    if (state._tag === "Ready") return yield* SubscriptionRef.set(reachability, "Connected")
    if (state._tag !== "Failed" && state._tag !== "Closed") return
    if ((yield* SubscriptionRef.get(reachability)) !== "Connected") return
    yield* SubscriptionRef.set(reachability, "Reconnecting")
    yield* attempt.pipe(
      Effect.retry(Schedule.spaced("2 seconds").pipe(Schedule.upTo("30 seconds"))),
      Effect.catchAll(() => SubscriptionRef.update(reachability, current => current === "Reconnecting" ? "Unreachable" : current)),
      Effect.forkScoped,
    )
  })), Effect.forkScoped)
  return reachability.changes
}))

/** One attempt to reach the service again, after checking whether it now asks for sign-in. */
export const useReconnectAttempt = (): Effect.Effect<void, unknown> => {
  const connection = useServiceConnection()
  const recheck = useSignInRecheck()
  return useMemo(() => recheck.pipe(Effect.flatMap(required => required ? Effect.void : connection.connect)), [connection, recheck])
}

export const useServiceObservation = (): ServiceObservation => {
  const session = useSession()
  const connection = useServiceConnection()
  const owned = Option.isSome(session.clientWindow)
  const snapshot = useAtomValue(session.application)
  const state = useAtomValue(useMemo(() => Atom.make(connection.changes), [connection]))
  const attempt = useReconnectAttempt()
  // A desktop window learns about the service from its owner, so only a browser retries on its own.
  const reachability = useAtomValue(useMemo(() => Atom.make(owned ? Stream.succeed<Reachability>("Connected") : browserReachability(connection, attempt)), [owned, connection, attempt]))
  if (owned) {
    return {
      service: Result.map(snapshot, value => {
        const service = value.service
        switch (service._tag) {
          case "Ready": return { _tag: "Ready" as const }
          case "Failed": return { _tag: "Failed" as const, message: service.message }
          case "CleanupFailed": return { _tag: "CleanupFailed" as const, message: service.message }
          default: return { _tag: "Starting" as const }
        }
      }),
      tray: Result.isSuccess(snapshot) && snapshot.value.owner._tag === "Desktop" ? Option.some(snapshot.value.owner.tray) : Option.none(),
      canRetry: true,
    }
  }
  return {
    service: Result.isSuccess(reachability) && reachability.value !== "Connected" ? Result.success<ServiceView>({ _tag: reachability.value })
      : Result.map(state, value => value._tag === "Ready" ? { _tag: "Ready" as const } : { _tag: "Starting" as const }),
    tray: Option.none(),
    canRetry: false,
  }
}
