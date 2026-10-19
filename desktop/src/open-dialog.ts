import type { OpenDialogOptions, OpenDialogReturnValue } from "electron"
import { Effect, Option } from "effect"
import { HostError, type OpenPathRequest } from "./desktop-rpc"

/** The native open dialog for a request: one folder, on every desktop OS. */
export const openDialogOptions = (request: OpenPathRequest): OpenDialogOptions => ({
  title: request.title,
  ...Option.match(request.defaultPath, { onNone: () => ({}), onSome: defaultPath => ({ defaultPath }) }),
  properties: ["openDirectory", "createDirectory", "dontAddToRecent"],
})

/** ChooseOpenPath for Electron main: shows the native dialog and answers the chosen path, or none when cancelled. */
export const chooseOpenPath = (show: (options: OpenDialogOptions) => Promise<Pick<OpenDialogReturnValue, "canceled" | "filePaths">>) =>
  (request: OpenPathRequest) => Effect.tryPromise({
    try: () => show(openDialogOptions(request)),
    catch: () => new HostError({ message: "Magnitude couldn’t open the file dialog. Try again." }),
  }).pipe(Effect.map(result => ({ path: result.canceled ? Option.none<string>() : Option.fromNullable(result.filePaths[0]) })))
