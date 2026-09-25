import { Effect, Schema } from "effect"
import { createRequire } from "node:module"

export class ForegroundContinuationFailed extends Schema.TaggedError<ForegroundContinuationFailed>()("ForegroundContinuationFailed", {}) {
  override get message() { return "The update was installed, but the foreground command could not continue. Run the command again." }
}

/** Replaces this process only after verified installation and before service admission. */
export const makeUnixProcessContinuation = (addon: string) => Effect.try({
  try: () => createRequire(import.meta.url)(addon) as {
      replaceProcess(executable: string, args: readonly string[], environment: readonly string[]): never
    },
  catch: () => new ForegroundContinuationFailed(),
}).pipe(Effect.map(native => ({
  replace: (executable: string, args: readonly string[], environment: Readonly<Record<string, string | undefined>>) => Effect.try({
    try: (): never => native.replaceProcess(executable, args, Object.entries(environment).flatMap(([key, value]) => value === undefined ? [] : [`${key}=${value}`])),
    catch: () => new ForegroundContinuationFailed(),
  }),
})))
