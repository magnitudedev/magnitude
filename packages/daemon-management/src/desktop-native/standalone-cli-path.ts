import { Effect } from "effect"
import { MAC_CLI_DIRECTORY, makeMacCliPath } from "./mac-cli-path"

/** The CLI-only command folder, `~/.magnitude/bin`, which the install script links and puts on PATH. */
export const STANDALONE_CLI_DIRECTORY = MAC_CLI_DIRECTORY

/**
 * Adds the CLI-only install's command folder to login and interactive shells on macOS and Linux, with the
 * same marked startup-file block the desktop's macOS command registration writes. It only edits that block;
 * it never touches other `magnitude` commands or PATH entries.
 */
export const installStandaloneCliPath = (options: { readonly home: string; readonly environment: Readonly<Record<string, string>>; readonly platform: NodeJS.Platform }) =>
  Effect.flatMap(makeMacCliPath(options.home, options.environment, options.platform !== "darwin"), path => path.install)
