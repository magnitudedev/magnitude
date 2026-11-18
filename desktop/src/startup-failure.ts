import { Cause, Option } from "effect"
import { DataFolderUnusable } from "@magnitudedev/daemon-management/desktop-native"

export const STARTUP_FAILED = "Quit Magnitude and open it again. If it still won’t start, reinstall the application. Your downloaded models and configuration are stored separately."

/**
 * What the startup error dialog says. Reinstalling can't fix an unusable data folder, so its failure says what can,
 * naming the folder; any other failure gets the general advice, its own error being for the log.
 */
export const startupFailureDetail = (cause: Cause.Cause<unknown>): string =>
  Option.match(Option.filter(Cause.failureOption(cause), (error): error is DataFolderUnusable => error instanceof DataFolderUnusable), {
    onNone: () => STARTUP_FAILED,
    onSome: error => error.reason,
  })
