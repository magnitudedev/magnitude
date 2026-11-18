import type { ReactNode } from "react"
import { Array, Option } from "effect"
import type { NetworkAccessSettings } from "@magnitudedev/sdk"
import { ErrorNotice, NoticeAction } from "./error-notice"

type NetworkReach = Pick<NetworkAccessSettings, "enabled" | "bind" | "reachableAddresses">

/** The address to give other devices: one the running service listens on, never one only saved for the next start. */
export const reachableAt = (settings: NetworkReach): Option.Option<string> => !settings.enabled ? Option.none()
  : Option.orElse(Option.filter(settings.bind, bind => settings.reachableAddresses.includes(bind)), () => Array.head(settings.reachableAddresses))

/** Why a restart is needed, from what the service serves now against what is saved. */
export const restartReason = (pending: { readonly storage: boolean; readonly network: Option.Option<NetworkReach> }): string => {
  if (pending.storage) return Option.isSome(pending.network) ? "Magnitude is still using the previous model folder and network settings." : "Magnitude is still using the previous model folder."
  const network = Option.getOrUndefined(pending.network)
  if (network?.enabled && network.reachableAddresses.length === 0) return "Network access starts when Magnitude restarts. Until then, only this computer can connect."
  if (network && !network.enabled && network.reachableAddresses.length > 0) return "Other devices can still connect until Magnitude restarts."
  return "Magnitude is still using the previous network settings."
}

/** Presentation only, and placed in the flow of whatever shows it, so it never covers other content. */
export function RestartNotice({ reason, canRestart, failed, restarting, onRestart, children }: {
  readonly reason: string
  readonly canRestart: boolean
  readonly failed: boolean
  readonly restarting: boolean
  readonly onRestart: () => void
  readonly children?: ReactNode
}) {
  return <ErrorNotice severity={failed ? "error" : "info"}
    title={failed ? "Magnitude couldn’t restart" : "Restart to apply your changes"}
    description={canRestart ? reason : `${reason} Restart the Magnitude server to apply them.`}
    actions={canRestart ? <NoticeAction disabled={restarting} onClick={onRestart}>Restart Magnitude</NoticeAction> : undefined}>
    {children}
  </ErrorNotice>
}
