import { Context, DateTime, Effect, Layer, Option, Schema } from "effect"
import { defineFSM } from "@magnitudedev/utils/fsm"
import { Database, decodeRow } from "./database"
import { InfrastructureFailure, LeaseId, Provider, RunId, RunPlan, RunRequest, RunResult } from "./domain"
import { RunProgress } from "./progress"
import { WorkId } from "./work-identity"
import { targets } from "./catalog"
import { planExecution, WorkSpec } from "./execution-plan"
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
  readonly progress: (id: RunId) => Effect.Effect<RunProgress, InfrastructureFailure | RunNotFound>
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
      const execution = yield* planExecution(plan.request, plan.targets, targets).pipe(Effect.mapError(error => new AdmissionRejected({ message: error.message })))
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
      for (const work of execution) {
        const spec = yield* Schema.encode(Schema.parseJson(WorkSpec))(work).pipe(Effect.mapError(() => new AdmissionRejected({ message: "Invalid execution work" })))
        yield* tx.query("INSERT INTO lab_work(run_id,work_id,state,spec,producer_id) VALUES($1,$2,'Queued',$3,$4)", [id, work.id, spec, work.kind === "test" ? Option.getOrNull(work.producer) : null])
      }
      yield* tx.query("INSERT INTO lab_events(run_id,kind,detail) VALUES($1,'submitted',$2)", [id, encoded])
      return yield* record(rows[0])
    })),
    get: id => Effect.gen(function* () {
      const rows = yield* db.query("SELECT * FROM lab_runs WHERE run_id=$1", [id])
      if (!rows[0]) return yield* new RunNotFound({ runId: id })
      return yield* record(rows[0])
    }),
    progress: id => db.transaction(tx => Effect.gen(function* () {
      const rows = yield* tx.query("SELECT * FROM lab_runs WHERE run_id=$1", [id])
      if (!rows[0]) return yield* new RunNotFound({ runId: id })
      const run = yield* record(rows[0])
      const stages = yield* tx.query("SELECT spec,state,attempts FROM lab_work WHERE run_id=$1 ORDER BY work_id", [id])
      const work = yield* decodeRow(Schema.Array(Schema.Struct({ spec: Schema.parseJson(WorkSpec), state: Schema.Literal("Queued", "Running", "Finished"), attempts: Schema.Int })), stages)
      const observed = yield* tx.query("SELECT work_id,lease_id,provider,resource_name,state FROM lab_leases WHERE run_id=$1 ORDER BY work_fence,lease_id", [id])
      const leases = yield* decodeRow(Schema.Array(Schema.Struct({ work_id: WorkId, lease_id: LeaseId, provider: Provider, resource_name: Schema.String,
        state: Schema.Literal("Allocating", "Ready", "Releasing", "Released") })), observed)
      return RunProgress.make({ runId: id, state: run.state._tag, deadline: run.deadline, stages: work.map(item => ({
        id: item.spec.id, kind: item.spec.kind, targetId: item.spec.target.target.id,
        backend: item.spec.kind === "build" ? item.spec.backend : item.spec.target.target.backend,
        state: item.state, attempts: item.attempts, producer: item.spec.kind === "test" ? item.spec.producer : Option.none(),
        leases: leases.filter(lease => lease.work_id === item.spec.id).map(lease => ({ id: lease.lease_id, provider: lease.provider, machine: lease.resource_name, state: lease.state })),
      })) })
    })),
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
