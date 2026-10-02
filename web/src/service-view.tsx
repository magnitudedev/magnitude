import { createContext, useContext, useMemo, type ReactNode } from "react"
import { Atom, Result, useAtomValue } from "@effect-atom/atom-react"
import { Option } from "effect"
import type { MagnitudeConnection } from "@magnitudedev/sdk"
import type { TrayRegistration } from "@magnitudedev/sdk/desktop-host"
import { useSession } from "./session"

/**
 * The service as this window can know it. A desktop window asks its owner, which still knows the
 * service when ACN is down; a browser only knows whether it can reach ACN.
 */
export type ServiceView =
  | { readonly _tag: "Ready" }
  | { readonly _tag: "Starting" }
  | { readonly _tag: "Failed"; readonly message: string }
  | { readonly _tag: "CleanupFailed"; readonly message: string }
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

export const useServiceObservation = (): ServiceObservation => {
  const session = useSession()
  const connection = useServiceConnection()
  const owned = Option.isSome(session.clientWindow)
  const snapshot = useAtomValue(session.application)
  const state = useAtomValue(useMemo(() => Atom.make(connection.changes), [connection]))
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
    service: Result.map(state, value => value._tag === "Ready" ? { _tag: "Ready" as const }
      : value._tag === "Failed" || value._tag === "Closed" ? { _tag: "Unreachable" as const }
      : { _tag: "Starting" as const }),
    tray: Option.none(),
    canRetry: false,
  }
}
