import { Data, Effect, Schema } from "effect"
import { BunSqliteDriver } from "@magnitudedev/daemon-management/bun"

class NativeRuntimeCheckFailed extends Data.TaggedError("NativeRuntimeCheckFailed")<{
  readonly message: string
}> {}

/** Release-only probe for the native database and Bun runtime retained by the headless CLI. */
export const runNativeRuntimeCheck = () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const database = yield* BunSqliteDriver.open(":memory:", { create: true })
  yield* database.execute("CREATE TABLE probe (value TEXT)")
  yield* database.execute("INSERT INTO probe VALUES (?)", ["Magnitude"])
  const rows = yield* database.query("SELECT value FROM probe")
  yield* Schema.decodeUnknown(Schema.Tuple(Schema.Struct({ value: Schema.Literal("Magnitude") })))(rows).pipe(
    Effect.mapError(() => new NativeRuntimeCheckFailed({ message: "Native database probe failed" })),
  )
  const result = yield* Effect.sync(() => {
    const hot = (value: number) => (value * 31 + 7) | 0
    let value = 0
    for (let index = 0; index < 1_000_000; index++) value = hot(value)
    return value
  })
  if (!Number.isInteger(result)) return yield* new NativeRuntimeCheckFailed({ message: "Bun runtime probe failed" })
  yield* Effect.try({
    try: () => process.stdout.write("Bun and SQLite native runtime ready\n"),
    catch: () => new NativeRuntimeCheckFailed({ message: "Unable to write native runtime check result" }),
  })
})))
