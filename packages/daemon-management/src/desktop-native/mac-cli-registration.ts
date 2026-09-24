import { Effect } from "effect"
import { join } from "node:path"
import { makeMacCliLink } from "./mac-cli-link"
import { MAC_CLI_DIRECTORY, makeMacCliPath } from "./mac-cli-path"

/** Desktop and installation share command placement and ownership of shell PATH edits. */
export const makeMacCliRegistration = (options: {
  readonly home: string
  readonly resourcesDirectory: string
  readonly environment: Readonly<Record<string, string>>
}) => Effect.gen(function* () {
  const link = yield* makeMacCliLink({ link: join(options.home, MAC_CLI_DIRECTORY, "magnitude"),
    target: join(options.resourcesDirectory, "magnitude"), path: options.environment.PATH ?? "" })
  const path = yield* makeMacCliPath(options.home, options.environment)
  const registration = yield* Effect.makeSemaphore(1)
  return {
    install: registration.withPermits(1)(link.install.pipe(Effect.zipRight(path.install))),
    remove: registration.withPermits(1)(Effect.gen(function* () {
      if ((yield* link.read) !== "Installed") return
      yield* path.remove
      yield* link.remove
    })),
  }
})
