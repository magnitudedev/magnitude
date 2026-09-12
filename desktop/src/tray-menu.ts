import type { OwnedServiceState } from "@magnitudedev/sdk/desktop-host"
import type { DesktopPage } from "@magnitudedev/client-common"

interface TrayActions {
  readonly open: (page?: DesktopPage) => void
  readonly stopModel: () => void
  readonly quit: () => void
}
export const buildTrayMenu = (state: {
  readonly service: OwnedServiceState["_tag"] | "Unknown"
  readonly model: { readonly label: string; readonly canStop: boolean }
  readonly setup: "Required" | "Complete" | "Unavailable"
}, actions: TrayActions) => [
  { label: state.service === "Ready" ? "Service running" : state.service === "Failed" || state.service === "CleanupFailed" ? "Service needs attention" : (state.service === "Stopping" || state.service === "Stopped") ? "Stopping Magnitude…" : "Service starting…", enabled: false },
  { label: state.service === "Ready" ? state.model.label : "Model status unavailable", enabled: false },
  ...(state.service === "Ready" && state.setup === "Required" ? [{ label: "Setup needed · Open Discover to begin", enabled: false }] : []),
  { type: "separator" as const },
  { label: "Open Magnitude", click: () => actions.open() },
  { label: "Discover Models", click: () => actions.open("discover") },
  { label: "Status", click: () => actions.open("status") },
  ...(state.service === "Ready" && state.model.canStop ? [{ label: "Stop Model", click: actions.stopModel }] : []),
  { type: "separator" as const },
  { label: "Quit Magnitude", click: actions.quit },
]
