import { Data, Effect, Option, Schema } from "effect"
import { ApplicationUpdateAction, type ApplicationUpdateState } from "@magnitudedev/sdk/desktop-host"
import { formatStorageSize } from "@magnitudedev/client-common"
import { updateApplication } from "../server/application"
import { isServerSetUp } from "../server/serve-owner"
import { runCommand } from "./output"

const failedPhrase = { verify: "could not be verified", install: "failed to install", authorization: "was not authorized to install", startup: "failed to start" } as const

/** `server` is a Headless owner running as the system service, which installs on its own when idle. */
export const renderApplicationUpdate = (state: ApplicationUpdateState, owner: "Desktop" | "Headless" | "None" = "Desktop", server = false): string => {
  // A running check supersedes the previous check's offer or failure; only transfers in progress stay visible.
  if (state.check._tag === "Checking" && (state.transfer._tag === "Idle" || state.transfer._tag === "Failed" || state.transfer._tag === "Available")) {
    return "Checking for application updates.\n"
  }
  switch (state.transfer._tag) {
    case "Unavailable": case "Failed": return `${state.transfer.message}\n`
    case "Available": {
      const offer = `Magnitude ${state.transfer.version} is available (${formatStorageSize(state.transfer.bytes)})`
      return Option.match(state.transfer.failed, {
        onNone: () => `${offer}.\nDownload: magnitude update download\n`,
        onSome: failed => `${offer}, but it ${failedPhrase[failed]} on this machine, so it is not downloaded automatically.\n`
          + `A newer release replaces it when one is published. To try it again anyway: magnitude update download\n`,
      })
    }
    case "Downloading": return `Downloading Magnitude ${state.transfer.version}: ${formatStorageSize(state.transfer.completed)} of ${formatStorageSize(state.transfer.total)}.\nCheck progress: magnitude update status\n`
    case "Staging": return `Preparing Magnitude ${state.transfer.version}.\nCheck progress: magnitude update status\n`
    case "InstallationFailed": return `${state.transfer.message}\nRetry installation: magnitude update install\nDiscard download: magnitude update discard\n`
    case "Ready": return `Magnitude ${state.transfer.version} is ready to install.\n${owner === "Headless"
      ? server ? "The server installs it on its own once it is idle." : "Stop `magnitude serve`, then run: magnitude update install"
      : owner === "None" ? "Install: magnitude update install" : "Install and restart: magnitude update install"}\n`
    case "Cancelling": return "Cancelling the automatic update download.\nCheck progress: magnitude update status\n"
    case "Closed": return "Magnitude is quitting.\n"
    case "Idle": return state.check._tag === "Succeeded" ? "Magnitude is up to date.\n"
      : state.check._tag === "Checking" ? "Checking for application updates.\n"
      : state.check._tag === "Failed" ? `${state.check.message}\n` : "No update check has completed.\nCheck now: magnitude update\n"
  }
}

/** Polls the update status until a check started by `update check` finishes, so its result is what is printed. */
export const awaitUpdateCheck = <A extends { readonly state: ApplicationUpdateState }, E, R>(started: A, status: Effect.Effect<A, E, R>) =>
  started.state.check._tag !== "Checking" ? Effect.succeed(started) : status.pipe(Effect.delay("250 millis"),
    Effect.repeat({ until: result => result.state.check._tag !== "Checking" }),
    Effect.timeout("2 minutes"), Effect.catchTag("TimeoutException", () => Effect.succeed(started)))

class UpdateCheckFailed extends Data.TaggedError("UpdateCheckFailed")<{ readonly message: string }> {}

/** A check's failure, when nothing else about the update is shown in its place. */
const failedUpdateCheck = (state: ApplicationUpdateState): Option.Option<string> =>
  state.transfer._tag === "Idle" && state.check._tag === "Failed" ? Option.some(state.check.message) : Option.none()

/**
 * Runs an update action through `update`. `install` with nothing downloaded reports the update status
 * instead of asking the owner to install, and a check that failed fails the command.
 */
export const performUpdate = <A extends { readonly state: ApplicationUpdateState }, E, R>(
  action: ApplicationUpdateAction,
  update: (action: ApplicationUpdateAction) => Effect.Effect<A, E, R>,
): Effect.Effect<A & { readonly installed: boolean }, E | UpdateCheckFailed, R> => {
  const performed = action === "install"
    ? update("status").pipe(Effect.flatMap(current => current.state.transfer._tag === "Idle"
      ? Effect.succeed({ ...current, installed: false }) : update("install").pipe(Effect.map(result => ({ ...result, installed: true })))))
    : update(action).pipe(
      Effect.flatMap(result => action === "check" ? awaitUpdateCheck(result, update("status")) : Effect.succeed(result)),
      Effect.map(result => ({ ...result, installed: false })))
  return performed.pipe(Effect.tap(result => Option.match(action === "check" || action === "install" ? failedUpdateCheck(result.state) : Option.none(), {
    onNone: () => Effect.void, onSome: message => new UpdateCheckFailed({ message }),
  })))
}

export const runUpdate = (input: string) => runCommand({
  effect: Schema.decodeUnknown(ApplicationUpdateAction)(input).pipe(Effect.flatMap(action => performUpdate(action, updateApplication).pipe(
    Effect.flatMap(result => (result.owner === "Headless" ? isServerSetUp : Effect.succeed(false)).pipe(Effect.map(server => ({ action, server, ...result }))))))),
  render: ({ action, state, owner, server, installed }) => installed ? owner === "None" ? "The Magnitude update was installed.\n" : "Magnitude is stopping its model and service to install the update and restart.\n" : action === "discard" ? "The prepared update was discarded.\n" : renderApplicationUpdate(state, owner, server),
})
