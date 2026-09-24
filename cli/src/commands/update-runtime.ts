import { Effect, Schema } from "effect"
import { ApplicationUpdateAction, type DesktopUpdateState } from "@magnitudedev/sdk/desktop-host"
import { formatStorageSize } from "@magnitudedev/client-common"
import { updateApplication } from "../server/application"
import { runCommand } from "./output"

export const renderApplicationUpdate = (state: DesktopUpdateState, owner: "Desktop" | "Headless" | "None" = "Desktop"): string => {
  if (state.check._tag === "Checking" && (state.transfer._tag === "Idle" || state.transfer._tag === "Failed")) {
    return "Checking for application updates.\n"
  }
  switch (state.transfer._tag) {
    case "Unavailable": case "Failed": return `${state.transfer.message}\n`
    case "Available": return `Magnitude ${state.transfer.version} is available (${formatStorageSize(state.transfer.bytes)}).\nDownload: magnitude update download\n`
    case "Downloading": return `Downloading Magnitude ${state.transfer.version}: ${formatStorageSize(state.transfer.completed)} of ${formatStorageSize(state.transfer.total)}.\nCheck progress: magnitude update status\n`
    case "Staging": return `Preparing Magnitude ${state.transfer.version}.\nCheck progress: magnitude update status\n`
    case "InstallationFailed": return `${state.transfer.message}\nRetry installation: magnitude update install\nDiscard download: magnitude update discard\n`
    case "Ready": return `Magnitude ${state.transfer.version} is ready to install.\n${owner === "Headless" ? "Stop the server, then run: magnitude serve" : owner === "None" ? "Install: magnitude update install" : "Install and restart: magnitude update install"}\n`
    case "Cancelling": return "Cancelling the automatic update download.\nCheck progress: magnitude update status\n"
    case "Closed": return "Magnitude is quitting.\n"
    case "Idle": return state.check._tag === "Succeeded" ? "Magnitude is up to date.\n"
      : state.check._tag === "Checking" ? "Checking for application updates.\n"
      : state.check._tag === "Failed" ? `${state.check.message}\n` : "No update check has completed.\nCheck now: magnitude update\n"
  }
}

export const runUpdate = (input: string) => runCommand({
  effect: Schema.decodeUnknown(ApplicationUpdateAction)(input).pipe(Effect.flatMap(action => updateApplication(action).pipe(Effect.map(result => ({ action, ...result }))))),
  render: ({ action, state, owner }) => action === "install" ? owner === "None" ? "The Magnitude update was installed.\n" : "Magnitude is stopping its model and service to install the update and restart.\n" : action === "discard" ? "The prepared update was discarded.\n" : renderApplicationUpdate(state, owner),
})
