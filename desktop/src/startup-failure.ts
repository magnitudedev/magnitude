import { Cause, Option } from "effect"
import { DataFolderUnusable } from "@magnitudedev/daemon-management/desktop-native"

export const STARTUP_FAILED = "Quit Magnitude and open it again. If it still won’t start, reinstall the application. Your downloaded models and configuration are stored separately."

export const RUNNING_FROM_DISK_IMAGE = "Magnitude is running from its disk image. Drag it to your Applications folder, then open it from there."

/** Whether a macOS app runs from a mounted disk image or a read-only copy macOS made of a downloaded app. */
export const runsFromDiskImage = (executable: string): boolean =>
  executable.startsWith("/Volumes/") || executable.includes("/AppTranslocation/")

/**
 * What the startup error dialog says. Reinstalling can't fix an unusable data folder, so its failure says what can,
 * naming the folder; an app opened from its disk image can't keep its installation there, so it says to move it;
 * any other failure gets the general advice, its own error being for the log.
 */
export const startupFailureDetail = (cause: Cause.Cause<unknown>, fromDiskImage: boolean): string =>
  Option.match(Option.filter(Cause.failureOption(cause), (error): error is DataFolderUnusable => error instanceof DataFolderUnusable), {
    onNone: () => fromDiskImage ? RUNNING_FROM_DISK_IMAGE : STARTUP_FAILED,
    onSome: error => error.reason,
  })
