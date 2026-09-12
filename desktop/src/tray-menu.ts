import type { OwnedServiceState } from "@magnitudedev/sdk/desktop-host"
import type { DesktopPage } from "@magnitudedev/client-common"

interface TrayActions {
  readonly open: (page?: DesktopPage) => void
  readonly stopModel: () => void
  readonly quit: () => void
  readonly restartUpdate: () => void
}
export const buildTrayMenu = (state: {
  readonly service: OwnedServiceState["_tag"] | "Unknown"
  readonly model: { readonly label: string; readonly canStop: boolean }
  readonly updateReady: boolean
}, actions: TrayActions) => [
  { label: state.service === "Ready" ? "Service running" : state.service === "Failed" || state.service === "CleanupFailed" ? "Service needs attention" : (state.service === "Stopping" || state.service === "Stopped") ? "Stopping Magnitude…" : "Service starting…", enabled: false },
  { label: state.service === "Ready" ? state.model.label : "Model status unavailable", enabled: false },
  { type: "separator" as const },
  { label: "Open Magnitude", click: () => actions.open() },
  { label: "Discover Models", click: () => actions.open("discover") },
  { label: "Status", click: () => actions.open("status") },
  ...(state.service === "Ready" && state.model.canStop ? [{ label: "Stop Model", click: actions.stopModel }] : []),
  { type: "separator" as const },
  ...(state.updateReady ? [{ label: "Restart to Update", click: actions.restartUpdate }] : []),
  { label: "Quit Magnitude", click: actions.quit },
]
