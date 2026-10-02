import type { MessageBoxOptions, MessageBoxReturnValue } from "electron"
import { Effect } from "effect"
import type { QuitFailureDecision } from "@magnitudedev/client-common/application/contracts"

/** The native fallback, used only when no window exists to show the app's own dialog. */
export const nativeQuitFailureDecision = (showDialog: (options: MessageBoxOptions) => Promise<MessageBoxReturnValue>) =>
  Effect.tryPromise(() => showDialog({
    type: "warning",
    title: "Magnitude could not finish quitting",
    message: "Background processes could not be confirmed stopped.",
    detail: "Retry Quit tries to stop background work again. Force Quit closes Magnitude even though some background processes may still be running.",
    buttons: ["Keep Magnitude Open", "Retry Quit", "Force Quit"],
    defaultId: 1,
    cancelId: 0,
    noLink: true,
  })).pipe(Effect.map(({ response }): QuitFailureDecision => response === 2 ? "ForceQuit" : response === 1 ? "RetryQuit" : "KeepOpen"))

/** A failed cleanup keeps ownership unless the user explicitly accepts an unproven exit. */
export const resolveQuitFailure = (message: string, host: {
  readonly decide: Effect.Effect<QuitFailureDecision, unknown>
  readonly forceQuit: () => void
}) => Effect.logError(message).pipe(Effect.zipRight(host.decide),
  Effect.flatMap(decision => decision === "ForceQuit"
    ? Effect.sync(() => { host.forceQuit(); return false })
    : Effect.succeed(decision === "RetryQuit")),
  Effect.catchAll(error => Effect.logError(error).pipe(Effect.as(false))),
)
