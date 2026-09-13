import { Effect, Schema } from "effect"
import { ApplicationUpdateAction, type DesktopUpdateState } from "@magnitudedev/sdk/desktop-host"
import { formatStorageSize } from "@magnitudedev/client-common"
import { updateDesktopApplication } from "../server/application"
import { runCommand } from "./output"

export const renderApplicationUpdate = (state: DesktopUpdateState): string => {
  switch (state.transfer._tag) {
    case "Unavailable": case "Failed": return `${state.transfer.message}\n`
    case "Available": return `Magnitude ${state.transfer.version} is available (${formatStorageSize(state.transfer.bytes)}).\nDownload: magnitude update download\n`
    case "Downloading": return `Downloading Magnitude ${state.transfer.version}: ${formatStorageSize(state.transfer.completed)} of ${formatStorageSize(state.transfer.total)}.\nCheck progress: magnitude update status\n`
    case "Staging": return `Preparing Magnitude ${state.transfer.version}.\nCheck progress: magnitude update status\n`
    case "Ready": return `Magnitude ${state.transfer.version} is ready to install.\nInstall and restart: magnitude update install\n`
    case "Cancelling": return "Cancelling the automatic update download.\nCheck progress: magnitude update status\n"
    case "Closed": return "Magnitude is quitting.\n"
    case "Idle": return state.check._tag === "Succeeded" ? "Magnitude is up to date.\n"
      : state.check._tag === "Checking" ? "Checking for application updates.\n"
      : state.check._tag === "Failed" ? `${state.check.message}\n` : "No update check has completed.\nCheck now: magnitude update\n"
  }
}

export const runUpdate = (input: string) => runCommand({
  effect: Schema.decodeUnknown(ApplicationUpdateAction)(input).pipe(Effect.flatMap(action => updateDesktopApplication(action).pipe(Effect.map(state => ({ action, state }))))),
  render: ({ action, state }) => action === "install" ? "Magnitude is stopping its model and service to install the update and restart.\n" : renderApplicationUpdate(state),
})
