import { Context, DateTime, Effect, Layer, Option, Schema } from "effect"
import { defineFSM } from "@magnitudedev/utils/fsm"
import { Database, decodeRow } from "./database"
import { CaseResult, InfrastructureFailure, RunId, RunPlan, RunResult, TargetId, TargetPlan } from "./domain"
import { Fence } from "./lease"
import { Cancelling as CancellingRun, Running as RunningRun, Queued as QueuedRun, runFSM } from "./run-store"

export const WorkClaim = Schema.Struct({ runId: RunId, targetId: TargetId, fence: Fence, worker: Schema.NonEmptyString })
export type WorkClaim = typeof WorkClaim.Type
export const WorkAssignment = Schema.Struct({ claim: WorkClaim, plan: RunPlan, target: TargetPlan, deadline: Schema.DateTimeUtc })
export type WorkAssignment = typeof WorkAssignment.Type
export const TargetResult = Schema.Struct({ cases: Schema.Array(CaseResult), cleanupErrors: Schema.Array(Schema.String) })
export type TargetResult = typeof TargetResult.Type
export class Queued extends Schema.TaggedClass<Queued>()("Queued", {}) {}
export class Running extends Schema.TaggedClass<Running>()("Running", {}) {}
export class Finished extends Schema.TaggedClass<Finished>()("Finished", {}) {}
export const workFSM = defineFSM({ Queued, Running, Finished }, {
  Queued: ["Running", "Finished"], Running: ["Queued", "Finished"], Finished: [],
})
export class StaleWork extends Schema.TaggedError<StaleWork>()("StaleWork", { runId: RunId, targetId: TargetId }) {}
export class InvalidResult extends Schema.TaggedError<InvalidResult>()("InvalidResult", { message: Schema.String }) {}
export interface WorkStore {
  readonly claim: (worker: string, seconds: number) => Effect.Effect<Option.Option<WorkAssignment>, InfrastructureFailure>
  readonly heartbeat: (claim: WorkClaim, seconds: number) => Effect.Effect<void, InfrastructureFailure | StaleWork>
  readonly finish: (claim: WorkClaim, result: TargetResult) => Effect.Effect<void, InfrastructureFailure | StaleWork | InvalidResult>
  readonly reconcile: () => Effect.Effect<void, InfrastructureFailure>
}
export const WorkStore = Context.GenericTag<WorkStore>("@magnitudedev/testing-lab/WorkStore")
const Candidate = Schema.Struct({ run_id: RunId, target_id: TargetId, plan: Schema.parseJson(RunPlan),
  run_state: Schema.Literal("Queued", "Running"), deadline: Schema.DateFromSelf, fence: Schema.NumberFromString.pipe(Schema.compose(Fence)) })
const duration = (seconds: number) => Number.isInteger(seconds) && seconds > 0 && seconds <= 3600
const invalidDuration = () => new InfrastructureFailure({ operation: "work-claim", message: "Claim duration must be 1–3600 seconds" })
const stale = (claim: WorkClaim) => new StaleWork({ runId: claim.runId, targetId: claim.targetId })
const key = (c: { readonly caseId: string; readonly harness: Option.Option<string> }) => `${c.caseId}/${Option.getOrElse(c.harness, () => "")}`
export const validateTargetResult = (target: typeof TargetPlan.Type, result: TargetResult) => Effect.gen(function* () {
  const expected = new Set(target.cases.map(c => key({ caseId: c.id, harness: c.harness })))
  const received = result.cases.map(key)
  if (received.length !== expected.size || new Set(received).size !== received.length ||
    result.cases.some(c => c.targetId !== target.target.id || !expected.has(key(c)) || c.outcome.status === "not-selected")) {
    return yield* new InvalidResult({ message: "Result must cover exactly the selected target, cases and harnesses" })
  }
})

export const WorkStoreLive = Layer.effect(WorkStore, Effect.gen(function* () {
  const db = yield* Database
  return {
    claim: (worker, seconds) => !duration(seconds) ? Effect.fail(invalidDuration()) : db.transaction(tx => Effect.gen(function* () {
      // Claim admission is short and serialized across schedulers so per-run concurrency is strict.
      yield* tx.query("SELECT pg_advisory_xact_lock(91826004)")
      const rows = yield* tx.query(`SELECT w.run_id,w.target_id,w.fence,r.plan,r.deadline,r.state AS run_state FROM lab_work w
        JOIN lab_runs r USING(run_id) WHERE w.state='Queued' AND r.state IN ('Queued','Running')
        AND r.deadline > clock_timestamp()
        AND (SELECT COUNT(*) FROM lab_work active WHERE active.run_id=r.run_id AND active.state='Running')
          < LEAST(4,(r.plan::jsonb->'request'->'limits'->>'concurrency')::int)
        ORDER BY r.created_at,w.target_id LIMIT 1 FOR UPDATE OF r,w SKIP LOCKED`)
      if (!rows[0]) return Option.none()
      const candidate = yield* decodeRow(Candidate, rows[0])
      const target = candidate.plan.targets.find(t => t.target.id === candidate.target_id)
      if (!target) return yield* new InfrastructureFailure({ operation: "work-claim", message: "Stored work has no matching target" })
      const state = workFSM.transition(new Queued({}), "Running", {})
      yield* tx.query(`UPDATE lab_work SET state=$3,worker=$4,claim_expires_at=LEAST(
        (SELECT deadline FROM lab_runs WHERE run_id=$1),clock_timestamp()+$5*interval '1 second'),attempts=attempts+1
        WHERE run_id=$1 AND target_id=$2`, [candidate.run_id, candidate.target_id, state._tag, worker, seconds])
      if (candidate.run_state === "Queued") {
        const run = runFSM.transition(new QueuedRun({ runId: candidate.run_id, plan: candidate.plan }), "Running", {})
        yield* tx.query("UPDATE lab_runs SET state=$2 WHERE run_id=$1", [candidate.run_id, run._tag])
      }
      yield* tx.query("INSERT INTO lab_attempts(run_id,target_id,fence,worker) VALUES($1,$2,$3,$4)", [candidate.run_id, candidate.target_id, candidate.fence, worker])
      return Option.some(WorkAssignment.make({ claim: { runId: candidate.run_id, targetId: candidate.target_id, fence: candidate.fence, worker }, plan: candidate.plan, target, deadline: DateTime.unsafeMake(candidate.deadline) }))
    })),
    heartbeat: (claim, seconds) => !duration(seconds) ? Effect.fail(invalidDuration()) : db.transaction(tx => Effect.gen(function* () {
      const rows = yield* tx.query(`UPDATE lab_work w SET claim_expires_at=LEAST(r.deadline,clock_timestamp()+$5*interval '1 second')
        FROM lab_runs r WHERE w.run_id=r.run_id AND w.run_id=$1 AND w.target_id=$2 AND w.fence=$3 AND w.worker=$4
        AND w.state='Running' AND w.claim_expires_at > clock_timestamp() AND r.state='Running' AND r.deadline > clock_timestamp()
        RETURNING w.fence`, [claim.runId, claim.targetId, claim.fence, claim.worker, seconds])
      if (!rows[0]) return yield* stale(claim)
    })),
    finish: (claim, result) => db.transaction(tx => Effect.gen(function* () {
      const rows = yield* tx.query(`SELECT r.plan FROM lab_runs r JOIN lab_work w USING(run_id)
        WHERE w.run_id=$1 AND w.target_id=$2 AND w.fence=$3 AND w.worker=$4 AND w.state='Running'
        AND w.claim_expires_at > clock_timestamp() AND r.state='Running' AND r.deadline > clock_timestamp()
        FOR UPDATE OF r,w`, [claim.runId, claim.targetId, claim.fence, claim.worker])
      if (!rows[0]) return yield* stale(claim)
      const row = yield* decodeRow(Schema.Struct({ plan: Schema.parseJson(RunPlan) }), rows[0])
      const target = row.plan.targets.find(t => t.target.id === claim.targetId)
      if (!target) return yield* new InvalidResult({ message: "Result target is absent from the run" })
      yield* validateTargetResult(target, result)
      const json = yield* Schema.encode(Schema.parseJson(TargetResult))(result).pipe(Effect.mapError(() => new InvalidResult({ message: "Malformed target result" })))
      const state = workFSM.transition(new Running({}), "Finished", {})
      yield* tx.query("UPDATE lab_work SET state=$3,result=$4 WHERE run_id=$1 AND target_id=$2", [claim.runId, claim.targetId, state._tag, json])
      yield* tx.query("UPDATE lab_attempts SET ended_at=clock_timestamp(),detail=$4 WHERE run_id=$1 AND target_id=$2 AND fence=$3", [claim.runId, claim.targetId, claim.fence, json])
      yield* tx.query("INSERT INTO lab_events(run_id,kind,detail) VALUES($1,'target-finished',$2)", [claim.runId, json])
    })),
    reconcile: () => db.transaction(tx => Effect.gen(function* () {
      yield* tx.query("SELECT pg_advisory_xact_lock(91826004)")
      const rows = yield* tx.query(`SELECT r.run_id,r.plan,r.state,r.created_at,
        r.deadline <= clock_timestamp() AS expired FROM lab_runs r WHERE state <> 'Finished' FOR UPDATE`)
      const runRows = yield* decodeRow(Schema.Array(Schema.Struct({ run_id: RunId, plan: Schema.parseJson(RunPlan),
        state: Schema.Literal("Queued", "Running", "Cancelling"), created_at: Schema.DateFromSelf, expired: Schema.Boolean })), rows)
      for (const run of runRows) {
        const pending = yield* tx.query(`SELECT w.target_id,w.state,w.fence,w.attempts FROM lab_work w
          WHERE w.run_id=$1 AND w.state <> 'Finished'
          AND ($2 OR $3 OR (w.state='Running' AND w.claim_expires_at <= clock_timestamp()))
          AND NOT EXISTS (SELECT 1 FROM lab_leases l WHERE l.run_id=w.run_id AND l.target_id=w.target_id AND l.state <> 'Released')
          FOR UPDATE`, [run.run_id, run.expired, run.state === "Cancelling"])
        const workRows = yield* decodeRow(Schema.Array(Schema.Struct({ target_id: TargetId, state: Schema.Literal("Queued", "Running"),
          fence: Schema.NumberFromString.pipe(Schema.compose(Fence)), attempts: Schema.Int })), pending)
        for (const work of workRows) {
          const target = run.plan.targets.find(t => t.target.id === work.target_id)
          if (!target) return yield* new InfrastructureFailure({ operation: "reconcile", message: "Work target absent from plan" })
          yield* tx.query(`UPDATE lab_attempts SET ended_at=clock_timestamp(),detail=$4 WHERE run_id=$1 AND target_id=$2 AND fence=$3 AND ended_at IS NULL`,
            [run.run_id, work.target_id, work.fence, "Worker claim expired or run was cancelled"])
          if (!run.expired && run.state !== "Cancelling" && work.state === "Running" && work.attempts < 2) {
            const next = workFSM.transition(new Running({}), "Queued", {})
            yield* tx.query("UPDATE lab_work SET state=$3,fence=fence+1,worker=NULL,claim_expires_at=NULL WHERE run_id=$1 AND target_id=$2", [run.run_id, work.target_id, next._tag])
            yield* tx.query("INSERT INTO lab_events(run_id,kind,detail) VALUES($1,'infrastructure-retry',$2)", [run.run_id, work.target_id])
            continue
          }
          const now = new Date().toISOString()
          const result: TargetResult = { cleanupErrors: [], cases: target.cases.map(c => ({ targetId: work.target_id, caseId: c.id,
            harness: c.harness, startedAt: now, endedAt: now, evidence: [], outcome: {
              status: run.state === "Cancelling" ? "cancelled" : "blocked",
              detail: run.state === "Cancelling" ? "Run cancelled" : run.expired ? "Run deadline expired" : "Worker disconnected after the permitted infrastructure retry",
            } })) }
          const json = yield* Schema.encode(Schema.parseJson(TargetResult))(result).pipe(Effect.orDie)
          const current = work.state === "Running" ? new Running({}) : new Queued({})
          const next = workFSM.transition(current, "Finished", {})
          yield* tx.query("UPDATE lab_work SET state=$3,fence=fence+1,result=$4 WHERE run_id=$1 AND target_id=$2", [run.run_id, work.target_id, next._tag, json])
        }
        const unfinished = yield* tx.query(`SELECT 1 FROM lab_work WHERE run_id=$1 AND state <> 'Finished'
          UNION ALL SELECT 1 FROM lab_leases WHERE run_id=$1 AND state <> 'Released' LIMIT 1`, [run.run_id])
        if (unfinished.length) continue
        const completed = yield* tx.query("SELECT result FROM lab_work WHERE run_id=$1 ORDER BY target_id", [run.run_id])
        const results = yield* decodeRow(Schema.Array(Schema.Struct({ result: Schema.parseJson(TargetResult) })), completed)
        const result = RunResult.make({ schemaVersion: 1, runId: run.run_id, plan: run.plan, startedAt: run.created_at.toISOString(), endedAt: new Date().toISOString(),
          cases: results.flatMap(r => r.result.cases), cleanupErrors: results.flatMap(r => r.result.cleanupErrors) })
        const encoded = yield* Schema.encode(Schema.parseJson(RunResult))(result).pipe(Effect.orDie)
        const props = { runId: run.run_id, plan: run.plan }
        const current = run.state === "Running" ? new RunningRun(props) : run.state === "Cancelling" ? new CancellingRun(props)
          : runFSM.transition(new QueuedRun(props), "Cancelling", {})
        const next = runFSM.transition(current, "Finished", {})
        yield* tx.query("UPDATE lab_runs SET state=$2,result=$3 WHERE run_id=$1", [run.run_id, next._tag, encoded])
        yield* tx.query("INSERT INTO lab_events(run_id,kind,detail) VALUES($1,'finished',$2)", [run.run_id, encoded])
      }
    })),
  } satisfies WorkStore
}))
