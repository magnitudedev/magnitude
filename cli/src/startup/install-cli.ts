import { BunContext } from "@effect/platform-bun"
import { installStandaloneCliPath } from "@magnitudedev/daemon-management/desktop-native"
import { Effect } from "effect"
import { homedir } from "node:os"

/** Run by install.sh from the unpacked CLI-only payload, after it links `~/.magnitude/bin/magnitude`. */
export const runCliInstallation = () => Effect.runPromise(Effect.gen(function* () {
  const environment = Object.fromEntries(Object.entries(process.env).filter((entry): entry is [string, string] => entry[1] !== undefined))
  yield* installStandaloneCliPath({ home: homedir(), environment })
}).pipe(Effect.provide(BunContext.layer),
  Effect.catchAll(error => Effect.sync(() => { process.stderr.write(`${error.message}\n`); process.exitCode = 1 }))))
