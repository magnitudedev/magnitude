import ansis from "ansis"
import { Effect } from "effect"

export const legacyCliNotice = `========================================================
MAGNITUDE HAS MOVED TO A FREE, OPEN SOURCE DESKTOP APP.

THIS LEGACY CLI WILL NO LONGER RECEIVE UPDATES.

DOWNLOAD THE DESKTOP APP:
https://magnitude.dev
========================================================`

export const showLegacyCliNotice = Effect.gen(function* () {
  const interactive = Boolean(process.stdin.isTTY && process.stdout.isTTY && process.stderr.isTTY)
  yield* Effect.sync(() => {
    const notice = interactive ? ansis.bold.yellow(legacyCliNotice) : legacyCliNotice
    process.stderr.write(`\n${notice}\n\n`)
  })
  if (interactive) yield* Effect.sleep("3 seconds")
})
