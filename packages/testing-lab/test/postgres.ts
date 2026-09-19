import { FileSystem } from "@effect/platform"
import { Effect, Option, Redacted } from "effect"
import { join } from "node:path"
import { databaseLayer } from "../src/database"
import { checkedCommand } from "../src/process"

export const temporaryDatabase = Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-postgres-" })
  const discovered = yield* checkedCommand("pg_config", ["--bindir"]).pipe(Effect.map(r => r.stdout.trim()), Effect.option)
  const bin = process.env.LAB_TEST_POSTGRES_BIN ?? Option.getOrElse(discovered, () => "/Applications/Postgres.app/Contents/Versions/latest/bin")
  const data = join(root, "data")
  yield* checkedCommand(join(bin, "initdb"), ["-D", data, "-U", "lab", "--auth=trust", "--no-locale"], { timeoutMs: 30_000 })
  yield* Effect.acquireRelease(
    checkedCommand(join(bin, "pg_ctl"), ["-D", data, "-l", join(root, "postgres.log"), "-o", `-k ${root} -h '' -F`, "-w", "start"]),
    () => checkedCommand(join(bin, "pg_ctl"), ["-D", data, "-m", "immediate", "-w", "stop"]).pipe(Effect.orDie),
  )
  return databaseLayer(Redacted.make(`postgresql://lab@localhost/postgres?host=${encodeURIComponent(root)}`))
})
