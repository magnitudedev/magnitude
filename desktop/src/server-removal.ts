import { Command } from "@effect/platform"
import { LINUX_SERVER_REMOVER_PATH } from "@magnitudedev/release/executables"

const MAC_REMOVAL_PROMPT = "Magnitude needs to remove the server set up on this Mac so the app can run. The server's data is kept."

/**
 * Removes the server through the platform's administrator prompt. On Linux, Polkit admits the person
 * at the machine to the removal wrapper without a password; macOS asks for an administrator.
 */
export const serverRemovalCommand = (platform: NodeJS.Platform, cli: string) => platform === "darwin"
  ? Command.make("/usr/bin/osascript", "-e",
    `do shell script quoted form of ${JSON.stringify(cli)} & " _server-remove" with prompt ${JSON.stringify(MAC_REMOVAL_PROMPT)} with administrator privileges`)
  // Any Polkit prompt, for a session that isn't the active one, comes from the desktop's own authentication agent only.
  : Command.make("/usr/bin/pkexec", "--disable-internal-agent", LINUX_SERVER_REMOVER_PATH)

/**
 * Why removal didn't happen, for a nonzero exit. A dismissed prompt is the person's choice, not a
 * missing administrator: osascript reports AppleScript's user-cancelled error -128, pkexec exits 126.
 */
export const serverRemovalFailure = (platform: NodeJS.Platform, code: number, stderr: string) =>
  (platform === "darwin" ? /\(-128\)/.test(stderr) : code === 126)
    ? "Removing the server was cancelled, so it is still running."
    : "The server was not removed. Removing it needs an administrator of this computer: try again as one, or ask one to run `magnitude server remove`."
