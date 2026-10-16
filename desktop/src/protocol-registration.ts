import { Effect } from "effect"
import { resolve } from "node:path"
import { DEEP_LINK_SCHEME } from "./deep-link"

export const DEVELOPMENT_PROTOCOL_VARIABLE = "MAGNITUDE_DESKTOP_REGISTER_PROTOCOL"

export interface ProtocolClientHost {
  readonly setAsDefaultProtocolClient: (protocol: string, path?: string, args?: string[]) => boolean
}

/**
 * Installed builds register `magnitude://` in their package (Info.plist, the installer's registry
 * keys, the `.desktop` entry). A development build registers itself only on request, because doing
 * so replaces the installed app as the handler for every link. Linux handlers are `.desktop` entries,
 * which a development checkout does not have.
 */
export const registerDevelopmentProtocolClient = (options: {
  readonly app: ProtocolClientHost
  readonly platform: NodeJS.Platform
  readonly environment: NodeJS.ProcessEnv
  readonly argv: ReadonlyArray<string>
  readonly execPath?: string
}) => Effect.gen(function* () {
  if (options.environment[DEVELOPMENT_PROTOCOL_VARIABLE] !== "1") return
  if (options.platform === "linux") return yield* Effect.logInfo("Development builds can't register magnitude:// links on Linux; install the package to test them.")
  // Electron started as `electron <app directory>` must be given the directory again to relaunch the app.
  const registered = options.platform === "win32" && options.argv[1] !== undefined
    ? options.app.setAsDefaultProtocolClient(DEEP_LINK_SCHEME, options.execPath ?? process.execPath, [resolve(options.argv[1])])
    : options.app.setAsDefaultProtocolClient(DEEP_LINK_SCHEME)
  if (registered) yield* Effect.logWarning(`This development build now opens ${DEEP_LINK_SCHEME}:// links. Reinstall Magnitude to restore the installed app as their handler.`)
  else yield* Effect.logWarning(`Could not register this development build for ${DEEP_LINK_SCHEME}:// links.`)
})
