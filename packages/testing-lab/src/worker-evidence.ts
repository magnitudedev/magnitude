import { Context, Effect, Layer, Option, Redacted, Schema, Stream } from "effect"
import { ArtifactStore } from "./artifact-store"
import { defineFSM } from "@magnitudedev/utils/fsm"
import { Database, decodeRow } from "./database"
import { Digest, InfrastructureFailure } from "./domain"
import { InvalidResult } from "./work-store"
import { WorkerAccessDenied, WorkerTickets } from "./worker-tickets"

const UploadId = Schema.UUID.pipe(Schema.brand("WorkerUploadId"))
class Uploading extends Schema.TaggedClass<Uploading>()("Uploading", {}) {}
class Verified extends Schema.TaggedClass<Verified>()("Verified", {}) {}
const uploadFSM = defineFSM({ Uploading, Verified }, { Uploading: ["Verified"], Verified: [] })
export const WorkerEvidenceLimits = { objectBytes: 256 * 1024 ** 2, attemptBytes: 4 * 1024 ** 3, objectCount: 1000 } as const
export const workerObjectLimits = (kind: "build" | "test") => kind === "build"
  ? { objectBytes: 4 * 1024 ** 3, attemptBytes: 16 * 1024 ** 3, objectCount: 1000 } : WorkerEvidenceLimits
export interface WorkerEvidence {
  readonly upload: (token: Redacted.Redacted<string>, digest: Digest, bytes: number, content: Stream.Stream<Uint8Array, InfrastructureFailure>) => Effect.Effect<void, WorkerAccessDenied | InvalidResult | InfrastructureFailure>
}
export const WorkerEvidence = Context.GenericTag<WorkerEvidence>("@magnitudedev/testing-lab/WorkerEvidence")

/** Reserve storage before consuming bytes; publish evidence authority only after successful integrity verification. */
export const WorkerEvidenceLive = Layer.effect(WorkerEvidence, Effect.gen(function* () {
  const tickets = yield* WorkerTickets
  const db = yield* Database
  const objects = yield* ArtifactStore
  return {
    upload: (token, digest, bytes, content) => Effect.gen(function* () {
      if (!Number.isSafeInteger(bytes) || bytes < 0 || bytes > 4 * 1024 ** 3) return yield* new InvalidResult({ message: "Invalid evidence upload length" })
      const reserve = tickets.withAuthority(token, (invocation, tx) => Effect.gen(function* () {
        const limits = workerObjectLimits(invocation.assignment.work.kind)
        if (bytes > limits.objectBytes) return yield* new InvalidResult({ message: "Object exceeds this work stage's upload limit" })
        const claim = invocation.assignment.claim
        const key = [claim.runId, claim.workId, claim.fence]
        yield* tx.query("DELETE FROM lab_worker_objects WHERE run_id=$1 AND work_id=$2 AND fence=$3 AND state='Uploading' AND upload_expires_at <= clock_timestamp()", key)
        const rows = yield* tx.query("SELECT bytes,state FROM lab_worker_objects WHERE run_id=$1 AND work_id=$2 AND fence=$3 AND digest=$4", [...key, digest])
        if (rows[0]) {
          const existing = yield* decodeRow(Schema.Struct({ bytes: Schema.NumberFromString, state: Schema.Literal("Uploading", "Verified") }), rows[0])
          if (existing.bytes !== bytes || existing.state !== "Verified") return yield* new InvalidResult({ message: "Evidence has a conflicting or active upload" })
          return { claim, upload: Option.none<typeof UploadId.Type>() }
        }
        if ((yield* tx.query("SELECT 1 FROM lab_worker_results WHERE run_id=$1 AND work_id=$2 AND fence=$3", key)).length) return yield* new InvalidResult({ message: "Cannot add evidence after result receipt" })
        const totals = yield* tx.query("SELECT count(*)::int AS count,COALESCE(sum(bytes),0)::text AS bytes FROM lab_worker_objects WHERE run_id=$1 AND work_id=$2 AND fence=$3", key)
        const total = yield* decodeRow(Schema.Struct({ count: Schema.Int, bytes: Schema.NumberFromString }), totals[0])
        if (total.count >= limits.objectCount || total.bytes + bytes > limits.attemptBytes) return yield* new InvalidResult({ message: "Attempt evidence budget exceeded" })
        const id = UploadId.make(crypto.randomUUID())
        yield* tx.query(`INSERT INTO lab_worker_objects(run_id,work_id,fence,digest,bytes,upload_id,upload_expires_at)
          VALUES($1,$2,$3,$4,$5,$6,clock_timestamp()+interval '15 minutes')`, [...key, digest, bytes, id])
        return { claim, upload: Option.some(id) }
      }))
      yield* Effect.acquireUseRelease(reserve, reservation => Effect.gen(function* () {
        let received = 0
        const checked = content.pipe(Stream.tap(chunk => Effect.gen(function* () {
          received += chunk.byteLength
          if (received > bytes) return yield* new InfrastructureFailure({ operation: "worker-evidence", message: "Evidence exceeds declared length" })
        })), Stream.concat(Stream.drain(Stream.fromEffect(Effect.suspend(() => received === bytes ? Effect.void
          : Effect.fail(new InfrastructureFailure({ operation: "worker-evidence", message: "Evidence is shorter than declared length" })))))))
        yield* objects.put(digest, checked)
        yield* tickets.withAuthority(token, (invocation, tx) => Effect.gen(function* () {
          const claim = invocation.assignment.claim
          if (Option.isSome(reservation.upload)) {
            const state = uploadFSM.transition(new Uploading({}), "Verified", {})
            const changed = yield* tx.query("UPDATE lab_worker_objects SET state=$2 WHERE upload_id=$1 AND state='Uploading' AND upload_expires_at > clock_timestamp() RETURNING digest", [reservation.upload.value, state._tag])
            if (!changed.length) return yield* new InvalidResult({ message: "Evidence upload reservation expired or was replaced" })
          }
          yield* tx.query("INSERT INTO lab_objects(owner,digest,bytes) VALUES($1,$2,$3) ON CONFLICT(owner,digest) DO NOTHING", [invocation.assignment.plan.request.owner, digest, bytes])
          if (claim.fence !== reservation.claim.fence || claim.runId !== reservation.claim.runId || claim.workId !== reservation.claim.workId) return yield* new WorkerAccessDenied({})
        }))
      }).pipe(Effect.timeoutFail({ duration: "10 minutes", onTimeout: () => new InfrastructureFailure({ operation: "worker-evidence", message: "Evidence upload timed out" }) })),
      reservation => Option.isSome(reservation.upload) ? db.query("DELETE FROM lab_worker_objects WHERE upload_id=$1 AND state='Uploading'", [reservation.upload.value]).pipe(Effect.orDie, Effect.asVoid) : Effect.void)
    }),
  } satisfies WorkerEvidence
}))
