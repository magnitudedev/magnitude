import { Context, DateTime, Effect, Layer, Option, Schema } from "effect"
import { defineFSM } from "@magnitudedev/utils/fsm"
import { Database, decodeRow } from "./database"
import { InfrastructureFailure, RunId, RunPlan, RunRequest, RunResult } from "./domain"
import { sha256 } from "./snapshot"

const identity = { runId: RunId, plan: RunPlan }
export class Queued extends Schema.TaggedClass<Queued>()("Queued", identity) {}
export class Running extends Schema.TaggedClass<Running>()("Running", identity) {}
export class Cancelling extends Schema.TaggedClass<Cancelling>()("Cancelling", identity) {}
export class Finished extends Schema.TaggedClass<Finished>()("Finished", identity) {}
export const runFSM = defineFSM({ Queued, Running, Cancelling, Finished }, {
  Queued: ["Running", "Cancelling"], Running: ["Cancelling", "Finished"], Cancelling: ["Finished"], Finished: [],
})
export const RunState = Schema.Union(Queued, Running, Cancelling, Finished)
export const RunRecord = Schema.Struct({ state: RunState, createdAt: Schema.DateTimeUtc, deadline: Schema.DateTimeUtc })
export type RunRecord = typeof RunRecord.Type
export class AdmissionRejected extends Schema.TaggedError<AdmissionRejected>()("AdmissionRejected", { message: Schema.String }) {}
export class RunNotFound extends Schema.TaggedError<RunNotFound>()("RunNotFound", { runId: RunId }) {}
export interface RunStore {
  readonly submit: (plan: RunPlan) => Effect.Effect<RunRecord, InfrastructureFailure | AdmissionRejected>
  readonly get: (id: RunId) => Effect.Effect<RunRecord, InfrastructureFailure | RunNotFound>
  readonly cancel: (id: RunId) => Effect.Effect<RunRecord, InfrastructureFailure | RunNotFound>
  readonly result: (id: RunId) => Effect.Effect<Option.Option<RunResult>, InfrastructureFailure | RunNotFound>
}
export const RunStore = Context.GenericTag<RunStore>("@magnitudedev/testing-lab/RunStore")
const Row = Schema.Struct({ run_id: RunId, state: Schema.Literal("Queued", "Running", "Cancelling", "Finished"),
  plan: Schema.parseJson(RunPlan), created_at: Schema.DateFromSelf, deadline: Schema.DateFromSelf,
  request_digest: Schema.String })
const record = (row: unknown) => decodeRow(Row, row).pipe(Effect.map(value => {
  const constructors = { Queued, Running, Cancelling, Finished }
  return RunRecord.make({ state: new constructors[value.state]({ runId: value.run_id, plan: value.plan }),
    createdAt: DateTime.unsafeMake(value.created_at), deadline: DateTime.unsafeMake(value.deadline) })
}))

/** Account cap covers all open reservations, not only currently booted machines. */
export const runStoreLayer = (accountBudgetUsd: number) => Layer.effect(RunStore, Effect.gen(function* () {
  const db = yield* Database
  return {
    submit: plan => db.transaction(tx => Effect.gen(function* () {
      const requestJson = yield* Schema.encode(Schema.parseJson(RunRequest))(plan.request).pipe(
        Effect.mapError(() => new AdmissionRejected({ message: "Invalid run request" })))
      const digest = sha256(requestJson)
      yield* tx.query("SELECT pg_advisory_xact_lock(91826003)")
      const existing = yield* tx.query("SELECT * FROM lab_runs WHERE owner=$1 AND idempotency_key=$2", [plan.request.owner, plan.request.idempotencyKey])
      if (existing[0]) {
        const decoded = yield* decodeRow(Row, existing[0])
        if (decoded.request_digest !== digest) return yield* new AdmissionRejected({ message: "Idempotency key already belongs to a different request" })
        return yield* record(existing[0])
      }
      const total = yield* tx.query("SELECT COALESCE(SUM(reserved_usd),0)::float8 AS value FROM lab_runs WHERE state <> 'Finished'")
      const amount = yield* decodeRow(Schema.Struct({ value: Schema.Number }), total[0])
      if (!Number.isFinite(accountBudgetUsd) || accountBudgetUsd <= 0 || !Number.isFinite(plan.estimatedComputeUsd) || plan.estimatedComputeUsd <= 0
        || plan.estimatedComputeUsd > plan.request.limits.budgetUsd || amount.value + plan.estimatedComputeUsd > accountBudgetUsd) {
        return yield* new AdmissionRejected({ message: "Run exceeds its compute budget or available account reservation" })
      }
      const encoded = yield* Schema.encode(Schema.parseJson(RunPlan))(plan).pipe(
        Effect.mapError(() => new AdmissionRejected({ message: "Invalid run plan" })))
      const id = RunId.make(`run-${crypto.randomUUID()}`)
      const rows = yield* tx.query(`INSERT INTO lab_runs(run_id,owner,idempotency_key,request_digest,plan,state,reserved_usd,deadline)
        VALUES($1,$2,$3,$4,$5,'Queued',$6,clock_timestamp()+$7*interval '1 minute') RETURNING *`,
      [id, plan.request.owner, plan.request.idempotencyKey, digest, encoded, plan.estimatedComputeUsd, plan.request.limits.deadlineMinutes])
      for (const target of plan.targets) {
        yield* tx.query("INSERT INTO lab_work(run_id,target_id,state) VALUES($1,$2,'Queued')", [id, target.target.id])
      }
      yield* tx.query("INSERT INTO lab_events(run_id,kind,detail) VALUES($1,'submitted',$2)", [id, encoded])
      return yield* record(rows[0])
    })),
    get: id => Effect.gen(function* () {
      const rows = yield* db.query("SELECT * FROM lab_runs WHERE run_id=$1", [id])
      if (!rows[0]) return yield* new RunNotFound({ runId: id })
      return yield* record(rows[0])
    }),
    result: id => Effect.gen(function* () {
      const rows = yield* db.query("SELECT result FROM lab_runs WHERE run_id=$1", [id])
      if (!rows[0]) return yield* new RunNotFound({ runId: id })
      const row = yield* decodeRow(Schema.Struct({ result: Schema.NullOr(Schema.parseJson(RunResult)) }), rows[0])
      return Option.fromNullable(row.result)
    }),
    cancel: id => db.transaction(tx => Effect.gen(function* () {
      const rows = yield* tx.query("SELECT * FROM lab_runs WHERE run_id=$1 FOR UPDATE", [id])
      if (!rows[0]) return yield* new RunNotFound({ runId: id })
      const run = yield* record(rows[0])
      if (run.state._tag === "Finished" || run.state._tag === "Cancelling") return run
      const state = runFSM.transition(run.state, "Cancelling", {})
      const changed = yield* tx.query("UPDATE lab_runs SET state=$2 WHERE run_id=$1 RETURNING *", [id, state._tag])
      // Increment every attempt fence in the same transaction as cancellation.
      yield* tx.query("UPDATE lab_attempts SET ended_at=clock_timestamp(),detail='Run cancelled' WHERE run_id=$1 AND ended_at IS NULL", [id])
      yield* tx.query("UPDATE lab_work SET fence=fence+1 WHERE run_id=$1 AND state <> 'Finished'", [id])
      yield* tx.query("INSERT INTO lab_events(run_id,kind,detail) VALUES($1,'cancelling','{}')", [id])
      return yield* record(changed[0])
    })),
  } satisfies RunStore
}))
