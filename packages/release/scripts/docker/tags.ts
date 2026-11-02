/** Prints the tags one release's image is published under, one per line: `tags.ts <repository> <version>`. */
import { BunRuntime } from "@effect/platform-bun"
import { Console, Effect } from "effect"
import { imageTags } from "./image"

const [repository, version] = process.argv.slice(2)

BunRuntime.runMain(Effect.gen(function* () {
  if (!repository || !version) return yield* Effect.dieMessage("Usage: tags.ts <repository> <version>")
  for (const tag of yield* imageTags(repository, version)) yield* Console.log(tag)
}))
