import { Effect, Schema } from "effect"
import { createRequire } from "node:module"

export class MacCliAuthorizationFailed extends Schema.TaggedError<MacCliAuthorizationFailed>()("MacCliAuthorizationFailed", {
  message: Schema.String,
}) {}

/** Invoked inside Electron so Authorization Services attributes the request to Magnitude. */
export const authorizeMacCliLink = (addonPath: string, link: string, target: string, remove: boolean) => Effect.tryPromise({
  try: () => {
    const native = createRequire(import.meta.url)(addonPath) as {
      configureCliLink: (link: string, target: string, remove: boolean) => Promise<void>
    }
    return native.configureCliLink(link, target, remove)
  },
  catch: error => new MacCliAuthorizationFailed({ message: error instanceof Error ? error.message : String(error) }),
})
