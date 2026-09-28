import type { MessageBoxOptions, MessageBoxReturnValue } from "electron"
import { Effect } from "effect"

/** A failed cleanup keeps ownership unless the user explicitly accepts an unproven exit. */
export const resolveQuitFailure = (message: string, host: {
  readonly showDialog: (options: MessageBoxOptions) => Promise<MessageBoxReturnValue>
  readonly forceQuit: () => void
}) => Effect.logError(message).pipe(Effect.zipRight(Effect.tryPromise(() => host.showDialog({
  type: "warning",
  title: "Magnitude could not finish quitting",
  message: "Background processes could not be confirmed stopped.",
  detail: "Retry Quit tries to stop background work again. Force Quit closes Magnitude even though some background processes may still be running.",
  buttons: ["Keep Magnitude Open", "Retry Quit", "Force Quit"],
  defaultId: 1,
  cancelId: 0,
  noLink: true,
}))),
  Effect.flatMap(({ response }) => response === 2
    ? Effect.sync(() => { host.forceQuit(); return false })
    : Effect.succeed(response === 1)),
  Effect.catchAll(error => Effect.logError(error).pipe(Effect.as(false))),
)
