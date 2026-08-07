import { appendFile } from "node:fs/promises"
import { BunContext } from "@effect/platform-bun"
import { Effect, Layer, Option, Schedule } from "effect"
import { ProcessStartIdentitySchema } from "../../acn-identity"
import { AcnRevisionSchema } from "../../acn-revision"
import { BunSqliteDriverLayer } from "../bun"
import { makeAcnOwnerStore } from "../owner-store"
import { makeAcnRevisionStore } from "../revision-store"

const [root, barrier, admissions] = process.argv.slice(2)
if (root === undefined || barrier === undefined || admissions === undefined) process.exit(2)

await Effect.runPromise(Effect.gen(function* () {
  const revisions = yield* makeAcnRevisionStore(root)
  const owners = yield* makeAcnOwnerStore(root)
  const revision = AcnRevisionSchema.make(1_000_001)
  const retryBusy = <A>(effect: Effect.Effect<A, { readonly _tag: string }>) => effect.pipe(
    Effect.retry({
      schedule: Schedule.intersect(Schedule.spaced("5 millis"), Schedule.recurs(100)),
      while: (error) => error._tag === "AcnProcessStoreBusy",
    }),
  )
  yield* retryBusy(revisions.register(revision))
  while (!(yield* Effect.promise(() => Bun.file(barrier).exists()))) {
    yield* Effect.sleep("5 millis")
  }
  const result = yield* owners.replaceOwner(Option.none(), {
    pid: process.pid,
    processStartIdentity: ProcessStartIdentitySchema.make(`fixture:${process.pid}`),
    port: 42_001 + (process.pid % 1_000),
  }, revision).pipe(retryBusy)
  if (result._tag === "Replaced") {
    yield* Effect.promise(() => appendFile(admissions, `${process.pid}\n`))
  }
}).pipe(Effect.provide(Layer.merge(BunContext.layer, BunSqliteDriverLayer))))
