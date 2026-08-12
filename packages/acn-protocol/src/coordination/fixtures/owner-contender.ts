import { appendFile } from "node:fs/promises"
import { BunContext } from "@effect/platform-bun"
import { Effect, Layer, Option } from "effect"
import { ProcessStartIdentitySchema } from "../../acn-identity"
import { BunSqliteDriverLayer } from "../bun"
import { makeAcnOwnerStore } from "../owner-store"

const [root, barrier, admissions] = process.argv.slice(2)
if (root === undefined || barrier === undefined || admissions === undefined) process.exit(2)

await Effect.runPromise(Effect.gen(function* () {
  const owners = yield* makeAcnOwnerStore(root)
  while (!(yield* Effect.promise(() => Bun.file(barrier).exists()))) {
    yield* Effect.sleep("5 millis")
  }
  const result = yield* owners.replaceOwner(Option.none(), {
    pid: process.pid,
    processStartIdentity: ProcessStartIdentitySchema.make(`fixture:${process.pid}`),
    port: 42_001 + (process.pid % 1_000),
  })
  if (result._tag === "Replaced") {
    yield* Effect.promise(() => appendFile(admissions, `${process.pid}\n`))
  }
}).pipe(Effect.provide(Layer.merge(BunContext.layer, BunSqliteDriverLayer))))
