import { writeFile } from "node:fs/promises"
import { BunContext } from "@effect/platform-bun"
import { Effect, Layer, Option } from "effect"
import { ProcessStartIdentitySchema } from "../../acn-identity"
import { makeAcnOwnerLock } from "../owner-lock"
import { BunSqliteMutexLayer } from "../bun"

const dataDir = process.argv[2]
const ready = process.argv[3]
if (dataDir === undefined || ready === undefined) {
  throw new Error("owner holder requires data root and ready paths")
}

await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const lock = yield* makeAcnOwnerLock(dataDir)
  const acquired = yield* lock.tryAcquire
  if (Option.isNone(acquired)) throw new Error("fixture could not acquire owner lock")
  yield* acquired.value.publish({
    pid: process.pid,
    processStartIdentity: ProcessStartIdentitySchema.make(`fixture:${process.pid}`),
    port: 42_001,
  })
  yield* Effect.tryPromise(() => writeFile(ready, "ready"))
  return yield* Effect.never
}).pipe(Effect.provide(Layer.merge(BunContext.layer, BunSqliteMutexLayer)))))
