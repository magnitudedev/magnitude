import { Context, Effect, Layer, Option, Redacted, Schema } from "effect"
import { Database, decodeRow } from "./database"
import { Digest, InfrastructureFailure } from "./domain"
import { sha256 } from "./snapshot"
import { InvalidResult, validateTargetResult, WorkClaim } from "./work-store"
import { WorkerReply } from "./worker-protocol"
import { WorkerAccessDenied, WorkerTickets } from "./worker-tickets"

export interface WorkerResults {
  readonly submit: (token: Redacted.Redacted<string>, reply: typeof WorkerReply.Type) => Effect.Effect<void, WorkerAccessDenied | InvalidResult | InfrastructureFailure>
  readonly read: (claim: WorkClaim) => Effect.Effect<Option.Option<typeof WorkerReply.Type>, InfrastructureFailure>
}
export const WorkerResults = Context.GenericTag<WorkerResults>("@magnitudedev/testing-lab/WorkerResults")
export const WorkerResultsLive = Layer.effect(WorkerResults, Effect.gen(function* () {
  const db = yield* Database
  const tickets = yield* WorkerTickets
  return {
    submit: (token, reply) => tickets.withAuthority(token, (invocation, tx) => Effect.gen(function* () {
      const assignment = invocation.assignment
      if (!Schema.equivalence(WorkClaim)(assignment.claim, reply.claim)) return yield* new InvalidResult({ message: "Worker reply belongs to another attempt" })
      yield* validateTargetResult(assignment.target, reply.result)
      const claim = reply.claim
      const rows = yield* tx.query("SELECT digest,bytes FROM lab_worker_objects WHERE run_id=$1 AND target_id=$2 AND fence=$3 AND state='Verified'", [claim.runId, claim.targetId, claim.fence])
      const objects = new Map((yield* Effect.forEach(rows, row => decodeRow(Schema.Struct({ digest: Digest, bytes: Schema.NumberFromString }), row))).map(row => [row.digest, row.bytes]))
      if (reply.result.cases.flatMap(test => test.evidence).some(item => objects.get(item.sha256) !== item.bytes)) return yield* new InvalidResult({ message: "Result references evidence not verified for this attempt" })
      const json = yield* Schema.encode(Schema.parseJson(WorkerReply))(reply)
      const digest = sha256(json)
      const existing = yield* tx.query("SELECT digest FROM lab_worker_results WHERE run_id=$1 AND target_id=$2 AND fence=$3", [claim.runId, claim.targetId, claim.fence])
      if (existing[0]) {
        if ((yield* decodeRow(Schema.Struct({ digest: Digest }), existing[0])).digest !== digest) return yield* new InvalidResult({ message: "A different result was already received for this attempt" })
        return
      }
      yield* tx.query("INSERT INTO lab_worker_results(run_id,target_id,fence,reply,digest) VALUES($1,$2,$3,$4,$5)", [claim.runId, claim.targetId, claim.fence, json, digest])
    })).pipe(Effect.mapError(error => error._tag === "ParseError" ? new InvalidResult({ message: "Malformed worker reply" }) : error)),
    read: claim => Effect.gen(function* () {
      const rows = yield* db.query("SELECT reply FROM lab_worker_results WHERE run_id=$1 AND target_id=$2 AND fence=$3", [claim.runId, claim.targetId, claim.fence])
      if (!rows[0]) return Option.none()
      const reply = (yield* decodeRow(Schema.Struct({ reply: Schema.parseJson(WorkerReply) }), rows[0])).reply
      if (!Schema.equivalence(WorkClaim)(claim, reply.claim)) return yield* new InfrastructureFailure({ operation: "worker-result", message: "Stored reply claim does not match reader" })
      return Option.some(reply)
    }),
  } satisfies WorkerResults
}))
