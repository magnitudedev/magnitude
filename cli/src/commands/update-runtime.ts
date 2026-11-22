import { Effect, Option, Schema } from "effect"
import { ApplicationUpdateAction, type ApplicationUpdateState } from "@magnitudedev/sdk/desktop-host"
import { formatStorageSize } from "@magnitudedev/client-common"
import { isStandaloneInstallation, updateApplication } from "../server/application"
import { runCommand } from "./output"
import { containerUpdateNotice } from "@magnitudedev/daemon-management/application-update"

const failedPhrase = { verify: "could not be verified", install: "failed to install", authorization: "was not authorized to install", startup: "failed to start" } as const

export const renderApplicationUpdate = (
  state: ApplicationUpdateState,
  owner: "Desktop" | "Headless" | "None" = "Desktop",
  standalone = false,
): string => {
  // A running check supersedes the previous check's offer or failure; only transfers in progress stay visible.
  if (state.check._tag === "Checking" && (state.transfer._tag === "Idle" || state.transfer._tag === "Failed" || state.transfer._tag === "Available" || state.transfer._tag === "ImageAvailable")) {
    return "Checking for updates.\n"
  }
  switch (state.transfer._tag) {
    case "Unavailable": case "Failed": return `${state.transfer.message}\n`
    case "Available": {
      const offer = `Magnitude ${state.transfer.version} is available (${formatStorageSize(state.transfer.bytes)})`
      const command = standalone ? "magnitude update install" : "magnitude update download"
      return Option.match(state.transfer.failed, {
        onNone: () => `${offer}.\n${standalone ? "Install" : "Download"}: ${command}\n`,
        onSome: failed => `${offer}, but it ${failedPhrase[failed]} on this machine, so it is not downloaded automatically.\n`
          + `A newer release replaces it when one is published. To try it again anyway: ${command}\n`,
      })
    }
    case "Downloading": return `Downloading Magnitude ${state.transfer.version}: ${formatStorageSize(state.transfer.completed)} of ${formatStorageSize(state.transfer.total)}.\nCheck progress: magnitude update status\n`
    case "Staging": return `Preparing Magnitude ${state.transfer.version}.\nCheck progress: magnitude update status\n`
    case "InstallationFailed": return `${state.transfer.message}\nRetry installation: magnitude update install\nDiscard download: magnitude update discard\n`
    case "Ready": return `Magnitude ${state.transfer.version} is ready to install.\n${owner === "Headless"
      ? "`magnitude serve` installs it once it is idle."
      : owner === "None" ? "Install: magnitude update install" : "Install and restart: magnitude update install"}\n`
    case "ImageAvailable": return `${containerUpdateNotice(state.transfer.version, state.transfer.image)}\n`
    case "Cancelling": return "Cancelling the automatic update download.\nCheck progress: magnitude update status\n"
    case "Closed": return "Magnitude is quitting.\n"
    case "Idle": return state.check._tag === "Succeeded" ? "Magnitude is up to date.\n"
      : state.check._tag === "Checking" ? "Checking for updates.\n"
      : state.check._tag === "Failed" ? `${state.check.message}\n` : "No update check has completed.\nCheck now: magnitude update\n"
  }
}

/** The reply to an install request a running `magnitude serve` accepts: it installs the update at its next idle point, once the download is verified. */
export const renderIdleInstallation = (state: ApplicationUpdateState, standalone = false): string => state.transfer._tag === "Ready"
  ? `Magnitude ${state.transfer.version} will be installed once \`magnitude serve\` is idle; it restarts into the new version.\nCheck progress: magnitude update status\n`
  : state.transfer._tag === "Downloading" || state.transfer._tag === "Staging"
  ? `Magnitude ${state.transfer.version} is downloading. Once it is verified, \`magnitude serve\` installs it when idle and restarts into the new version.\nCheck progress: magnitude update status\n`
  : renderApplicationUpdate(state, "Headless", standalone)

/** Polls the update status until a check started by `update check` finishes, so its result is what is printed. */
export const awaitUpdateCheck = <A extends { readonly state: ApplicationUpdateState }, E, R>(started: A, status: Effect.Effect<A, E, R>) =>
  started.state.check._tag !== "Checking" ? Effect.succeed(started) : status.pipe(Effect.delay("250 millis"),
    Effect.repeat({ until: result => result.state.check._tag !== "Checking" }),
    Effect.timeout("2 minutes"), Effect.catchTag("TimeoutException", () => Effect.succeed(started)))

export const runUpdate = (input: string) => runCommand({
  effect: Schema.decodeUnknown(ApplicationUpdateAction)(input).pipe(Effect.flatMap(action =>
    Effect.all({
      standalone: isStandaloneInstallation,
      result: updateApplication(action).pipe(
        Effect.flatMap(result => action === "check" ? awaitUpdateCheck(result, updateApplication("status")) : Effect.succeed(result)),
      ),
    }).pipe(Effect.map(({ standalone, result }) => ({ action, standalone, ...result }))))),
  render: ({ action, standalone, state, owner }) => action === "install" ? owner === "None" ? "The Magnitude update was installed.\n"
    : owner === "Headless" ? renderIdleInstallation(state, standalone)
    : "Magnitude is stopping its model and service to install the update and restart.\n"
    : action === "discard" ? "The prepared update was discarded.\n" : renderApplicationUpdate(state, owner, standalone),
})
