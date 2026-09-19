import { Context, DateTime, Effect, Layer, Redacted, Schema } from "effect"
import { randomBytes } from "node:crypto"
import { Database, decodeRow } from "./database"
import { InfrastructureFailure, RunPlan } from "./domain"
import { sha256 } from "./snapshot"
import { WorkAssignment } from "./work-store"
import { WorkerInvocation } from "./worker-protocol"

export const WorkerTicketId = Schema.UUID.pipe(Schema.brand("WorkerTicketId"))
export const WorkerTicket = Schema.Struct({ id: WorkerTicketId, token: Schema.Redacted(Schema.NonEmptyString) })
export class WorkerAccessDenied extends Schema.TaggedError<WorkerAccessDenied>()("WorkerAccessDenied", {}) {}
export interface WorkerTickets {
  readonly issue: (invocation: typeof WorkerInvocation.Type) => Effect.Effect<typeof WorkerTicket.Type, InfrastructureFailure | WorkerAccessDenied>
  readonly authorize: (token: Redacted.Redacted<string>) => Effect.Effect<typeof WorkerInvocation.Type, InfrastructureFailure | WorkerAccessDenied>
  readonly revoke: (id: typeof WorkerTicketId.Type) => Effect.Effect<void, InfrastructureFailure>
}
export const WorkerTickets = Context.GenericTag<WorkerTickets>("@magnitudedev/testing-lab/WorkerTickets")

/** Attempt-scoped credentials: the database stores only a digest; each access rechecks the live work fence. */
export const WorkerTicketsLive = Layer.effect(WorkerTickets, Effect.gen(function* () {
  const db = yield* Database
  return {
    issue: invocation => db.transaction(tx => Effect.gen(function* () {
      const claim = invocation.assignment.claim
      const job = invocation.assignment
      if (job.target.blockers.length || job.plan.request.trust === "untrusted-ci" &&
        (!invocation.disposable || job.target.target.provider === "local" || job.target.target.provider === "spark")) return yield* new WorkerAccessDenied({})
      const rows = yield* tx.query(`SELECT r.plan,r.deadline FROM lab_work w JOIN lab_runs r USING(run_id)
        WHERE w.run_id=$1 AND w.target_id=$2 AND w.fence=$3 AND w.worker=$4
        AND w.state='Running' AND w.claim_expires_at > clock_timestamp()
        AND r.state='Running' AND r.deadline > clock_timestamp() FOR UPDATE OF r,w`,
      [claim.runId, claim.targetId, claim.fence, claim.worker])
      if (!rows[0]) return yield* new WorkerAccessDenied({})
      const row = yield* decodeRow(Schema.Struct({ plan: Schema.parseJson(RunPlan), deadline: Schema.DateFromSelf }), rows[0])
      const target = row.plan.targets.find(target => target.target.id === claim.targetId)
      if (!target || !Schema.equivalence(WorkAssignment)(invocation.assignment,
        { claim, plan: row.plan, target, deadline: DateTime.unsafeMake(row.deadline) })) return yield* new WorkerAccessDenied({})
      const id = WorkerTicketId.make(crypto.randomUUID())
      const token = Redacted.make(randomBytes(32).toString("base64url"))
      const inserted = yield* tx.query(`INSERT INTO lab_worker_tickets(ticket_id,token_digest,run_id,target_id,fence,worker,invocation)
        VALUES($1,$2,$3,$4,$5,$6,$7) ON CONFLICT(run_id,target_id,fence) DO NOTHING RETURNING ticket_id`,
      [id, sha256(Redacted.value(token)), claim.runId, claim.targetId, claim.fence, claim.worker,
        yield* Schema.encode(Schema.parseJson(WorkerInvocation))(invocation)])
      // A lost issuer response cannot rotate a credential under an already executing guest.
      if (!inserted[0]) return yield* new WorkerAccessDenied({})
      return WorkerTicket.make({ id, token })
    })).pipe(Effect.mapError(error => error._tag === "ParseError" ? new InfrastructureFailure({ operation: "worker-ticket", message: "Cannot encode worker invocation" }) : error)),
    authorize: token => Effect.gen(function* () {
      if (!/^[A-Za-z0-9_-]{43}$/.test(Redacted.value(token))) return yield* new WorkerAccessDenied({})
      const rows = yield* db.query(`SELECT t.invocation FROM lab_worker_tickets t
        JOIN lab_work w ON w.run_id=t.run_id AND w.target_id=t.target_id
        JOIN lab_runs r ON r.run_id=t.run_id
        WHERE t.token_digest=$1 AND NOT t.revoked AND w.fence=t.fence AND w.worker=t.worker
        AND w.state='Running' AND w.claim_expires_at > clock_timestamp()
        AND r.state='Running' AND r.deadline > clock_timestamp()`, [sha256(Redacted.value(token))])
      if (!rows[0]) return yield* new WorkerAccessDenied({})
      return (yield* decodeRow(Schema.Struct({ invocation: Schema.parseJson(WorkerInvocation) }), rows[0])).invocation
    }),
    revoke: id => db.query("UPDATE lab_worker_tickets SET revoked=true WHERE ticket_id=$1", [id]).pipe(Effect.asVoid),
  } satisfies WorkerTickets
}))
