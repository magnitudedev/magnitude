import { appendFile } from "node:fs/promises"
import { BunContext } from "@effect/platform-bun"
import { Duration, Effect, Layer, Option } from "effect"
import { makeAcnOwnerLock } from "../owner-lock"
import { BunSqliteMutexLayer } from "../bun"

const dataDir = process.argv[2]
const barrier = process.argv[3]
const admissions = process.argv[4]
if (dataDir === undefined || barrier === undefined || admissions === undefined) {
  throw new Error("owner contender requires data root, barrier, and admissions paths")
}

while (!(await Bun.file(barrier).exists())) await Bun.sleep(1)

await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const lock = yield* makeAcnOwnerLock(dataDir)
  const acquired = yield* lock.tryAcquire
  if (Option.isNone(acquired)) return
  yield* Effect.tryPromise(() => appendFile(admissions, `${process.pid}\n`))
  yield* Effect.sleep(Duration.millis(200))
}).pipe(Effect.provide(Layer.merge(BunContext.layer, BunSqliteMutexLayer)))))
