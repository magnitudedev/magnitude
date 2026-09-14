import { NodeContext } from "@effect/platform-node"
import { Effect, Layer, Option } from "effect"
import { homedir } from "node:os"
import { cliBinaryResolverLayer } from "./cli-binary-resolver"
import { CliProcessSpawner, cliProcessSpawnerLayer } from "./cli-process-spawner"

const live = cliProcessSpawnerLayer({ args: process.argv.slice(2), environment: process.env }).pipe(
  Layer.provide(cliBinaryResolverLayer({
    platform: process.platform,
    home: homedir(),
    application: Option.fromNullable(process.env.MAGNITUDE_DESKTOP_PATH),
    localAppData: Option.fromNullable(process.env.LOCALAPPDATA),
  })),
  Layer.provide(NodeContext.layer),
)

const main = CliProcessSpawner.pipe(
  Effect.flatMap(spawner => spawner.spawn),
  Effect.provide(live),
  Effect.catchAll(error => Effect.sync(() => {
    console.error(error.reason)
    return 1
  })),
)
void Effect.runPromise(main).then(code => { process.exitCode = code })
