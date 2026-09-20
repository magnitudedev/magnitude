import { ArtifactStore } from "./artifact-store"
import { admitBuildOutput } from "./build-admission"
import { Context, DateTime, Effect, Layer, Option, Schema } from "effect"
import { defineFSM } from "@magnitudedev/utils/fsm"
import { Database, decodeRow } from "./database"
import { CaseResult, InfrastructureFailure, Input, RunId, RunPlan, RunResult, TargetId, TargetPlan } from "./domain"
import { WorkId, WorkSpec } from "./execution-plan"
import { BuildOutput } from "./build-output"
import { Fence } from "./lease"
import { Cancelling as CancellingRun, Running as RunningRun, Queued as QueuedRun, runFSM } from "./run-store"

export const WorkClaim = Schema.Struct({ runId: RunId, workId: WorkId, targetId: TargetId, fence: Fence, worker: Schema.NonEmptyString })
export type WorkClaim = typeof WorkClaim.Type
export const WorkAssignment = Schema.Struct({ claim: WorkClaim, plan: RunPlan, work: WorkSpec, input: Input, target: TargetPlan, deadline: Schema.DateTimeUtc })
export type WorkAssignment = typeof WorkAssignment.Type
export const TargetResult = Schema.Struct({ cases: Schema.Array(CaseResult), cleanupErrors: Schema.Array(Schema.String) })
export type TargetResult = typeof TargetResult.Type
export const WorkResult = Schema.Struct({ ...TargetResult.fields, output: Schema.optionalWith(BuildOutput, { as: "Option", exact: true }) })
export type WorkResult = typeof WorkResult.Type
export const assignmentInputs = (assignment: WorkAssignment) => [assignment.input, ...(assignment.work.kind === "test" ? Option.toArray(assignment.plan.request.updateFrom) : [])]
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
  readonly finish: (claim: WorkClaim, result: WorkResult) => Effect.Effect<void, InfrastructureFailure | StaleWork | InvalidResult>
  readonly reconcile: () => Effect.Effect<void, InfrastructureFailure>
}
export const WorkStore = Context.GenericTag<WorkStore>("@magnitudedev/testing-lab/WorkStore")
const Candidate = Schema.Struct({ run_id: RunId, work_id: WorkId, spec: Schema.parseJson(WorkSpec), output: Schema.NullOr(Schema.parseJson(BuildOutput)), plan: Schema.parseJson(RunPlan),
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
  const objects = yield* Effect.serviceOption(ArtifactStore)
  return {
    claim: (worker, seconds) => !duration(seconds) ? Effect.fail(invalidDuration()) : db.transaction(tx => Effect.gen(function* () {
      // Claim admission is short and serialized across schedulers so per-run concurrency is strict.
      yield* tx.query("SELECT pg_advisory_xact_lock(91826004)")
      const rows = yield* tx.query(`SELECT w.run_id,w.work_id,w.spec,p.output,w.fence,r.plan,r.deadline,r.state AS run_state FROM lab_work w
        JOIN lab_runs r ON r.run_id=w.run_id LEFT JOIN lab_work p ON p.run_id=w.run_id AND p.work_id=w.producer_id
        WHERE w.state='Queued' AND (w.producer_id IS NULL OR (p.state='Finished' AND p.output IS NOT NULL
          AND NOT EXISTS(SELECT 1 FROM lab_leases l WHERE l.run_id=p.run_id AND l.work_id=p.work_id AND l.state <> 'Released'))) AND r.state IN ('Queued','Running')
        AND r.deadline > clock_timestamp()
        AND (SELECT COUNT(*) FROM lab_work active WHERE active.run_id=r.run_id AND active.state='Running')
          < LEAST(4,(r.plan::jsonb->'request'->'limits'->>'concurrency')::int)
        ORDER BY r.created_at,w.work_id LIMIT 1 FOR UPDATE OF r,w SKIP LOCKED`)
      if (!rows[0]) return Option.none()
      const candidate = yield* decodeRow(Candidate, rows[0])
      const work = candidate.spec
      const target = work.target
      if (!target) return yield* new InfrastructureFailure({ operation: "work-claim", message: "Stored work has no matching target" })
      const state = workFSM.transition(new Queued({}), "Running", {})
      yield* tx.query(`UPDATE lab_work SET state=$3,worker=$4,claim_expires_at=LEAST(
        (SELECT deadline FROM lab_runs WHERE run_id=$1),clock_timestamp()+$5*interval '1 second'),attempts=attempts+1
        WHERE run_id=$1 AND work_id=$2`, [candidate.run_id, candidate.work_id, state._tag, worker, seconds])
      if (candidate.run_state === "Queued") {
        const run = runFSM.transition(new QueuedRun({ runId: candidate.run_id, plan: candidate.plan }), "Running", {})
        yield* tx.query("UPDATE lab_runs SET state=$2 WHERE run_id=$1", [candidate.run_id, run._tag])
      }
      yield* tx.query("INSERT INTO lab_attempts(run_id,work_id,fence,worker) VALUES($1,$2,$3,$4)", [candidate.run_id, candidate.work_id, candidate.fence, worker])
      return Option.some(WorkAssignment.make({ claim: { runId: candidate.run_id, workId: candidate.work_id, targetId: target.target.id, fence: candidate.fence, worker }, plan: candidate.plan, work, input: candidate.output ? { kind: "artifacts", digest: candidate.output.artifactDigest } : candidate.plan.request.input, target, deadline: DateTime.unsafeMake(candidate.deadline) }))
    })),
    heartbeat: (claim, seconds) => !duration(seconds) ? Effect.fail(invalidDuration()) : db.transaction(tx => Effect.gen(function* () {
      const rows = yield* tx.query(`UPDATE lab_work w SET claim_expires_at=LEAST(r.deadline,clock_timestamp()+$5*interval '1 second')
        FROM lab_runs r WHERE w.run_id=r.run_id AND w.run_id=$1 AND w.work_id=$2 AND w.fence=$3 AND w.worker=$4
        AND w.state='Running' AND w.claim_expires_at > clock_timestamp() AND r.state='Running' AND r.deadline > clock_timestamp()
        RETURNING w.fence`, [claim.runId, claim.workId, claim.fence, claim.worker, seconds])
      if (!rows[0]) return yield* stale(claim)
    })),
    finish: (claim, result) => db.transaction(tx => Effect.gen(function* () {
      const rows = yield* tx.query(`SELECT w.spec,r.plan FROM lab_runs r JOIN lab_work w USING(run_id)
        WHERE w.run_id=$1 AND w.work_id=$2 AND w.fence=$3 AND w.worker=$4 AND w.state='Running'
        AND w.claim_expires_at > clock_timestamp() AND r.state='Running' AND r.deadline > clock_timestamp()
        FOR UPDATE OF r,w`, [claim.runId, claim.workId, claim.fence, claim.worker])
      if (!rows[0]) return yield* stale(claim)
      const row = yield* decodeRow(Schema.Struct({ spec: Schema.parseJson(WorkSpec), plan: Schema.parseJson(RunPlan) }), rows[0])
      const target = row.spec.target
      if (claim.targetId !== target.target.id || claim.workId !== row.spec.id) return yield* new InvalidResult({ message: "Result target is absent from the run" })
      yield* validateTargetResult(target, result)
      if (row.spec.kind === "test" && Option.isSome(result.output)) return yield* new InvalidResult({ message: "A test consumer cannot publish build output" })
      if (row.spec.kind === "build" && result.cases.every(test => test.outcome.status === "passed") && Option.isNone(result.output)) {
        return yield* new InvalidResult({ message: "Successful build must publish an admitted package graph" })
      }
      let accepted = result, published: string | null = null
      if (Option.isSome(result.output)) {
        if (Option.isNone(objects)) return yield* new InvalidResult({ message: "Build admission requires artifact storage" })
        const receipt = yield* admitBuildOutput(tx, claim, row.plan, row.spec, result).pipe(
          Effect.provideService(ArtifactStore, objects.value), Effect.mapError(error => new InvalidResult({ message: error.message })))
        if (Option.isSome(receipt.published)) published = yield* Schema.encode(Schema.parseJson(BuildOutput))(receipt.published.value).pipe(Effect.orDie)
        accepted = { ...result, cases: result.cases.map(test => ({ ...test, evidence: [...test.evidence, ...Option.toArray(receipt.evidence)] })) }
      }
      const json = yield* Schema.encode(Schema.parseJson(WorkResult))(accepted).pipe(Effect.mapError(() => new InvalidResult({ message: "Malformed target result" })))
      const state = workFSM.transition(new Running({}), "Finished", {})
      const completed = yield* tx.query(`UPDATE lab_work SET state=$3,result=$4,output=$5 WHERE run_id=$1 AND work_id=$2
        AND claim_expires_at > clock_timestamp() AND EXISTS(SELECT 1 FROM lab_runs WHERE run_id=$1 AND deadline > clock_timestamp() AND state='Running') RETURNING work_id`,
        [claim.runId, claim.workId, state._tag, json, published])
      if (!completed.length) return yield* stale(claim)
      yield* tx.query("UPDATE lab_attempts SET ended_at=clock_timestamp(),detail=$4 WHERE run_id=$1 AND work_id=$2 AND fence=$3", [claim.runId, claim.workId, claim.fence, json])
      yield* tx.query("INSERT INTO lab_events(run_id,kind,detail) VALUES($1,'target-finished',$2)", [claim.runId, json])
    })),
    reconcile: () => db.transaction(tx => Effect.gen(function* () {
      yield* tx.query("SELECT pg_advisory_xact_lock(91826004)")
      const rows = yield* tx.query(`SELECT r.run_id,r.plan,r.state,r.created_at,
        r.deadline <= clock_timestamp() AS expired FROM lab_runs r WHERE state <> 'Finished' FOR UPDATE`)
      const runRows = yield* decodeRow(Schema.Array(Schema.Struct({ run_id: RunId, plan: Schema.parseJson(RunPlan),
        state: Schema.Literal("Queued", "Running", "Cancelling"), created_at: Schema.DateFromSelf, expired: Schema.Boolean })), rows)
      for (const run of runRows) {
        for (;;) {
        const pending = yield* tx.query(`SELECT w.work_id,w.spec,w.state,w.fence,w.attempts FROM lab_work w
          WHERE w.run_id=$1 AND w.state <> 'Finished'
          AND ($2 OR $3 OR (w.state='Running' AND w.claim_expires_at <= clock_timestamp())
            OR (w.state='Queued' AND EXISTS(SELECT 1 FROM lab_work p WHERE p.run_id=w.run_id AND p.work_id=w.producer_id AND p.state='Finished' AND p.output IS NULL)))
          AND NOT EXISTS (SELECT 1 FROM lab_leases l WHERE l.run_id=w.run_id AND l.work_id=w.work_id AND l.state <> 'Released')
          FOR UPDATE`, [run.run_id, run.expired, run.state === "Cancelling"])
        if (!pending.length) break
        const workRows = yield* decodeRow(Schema.Array(Schema.Struct({ work_id: WorkId, spec: Schema.parseJson(WorkSpec), state: Schema.Literal("Queued", "Running"),
          fence: Schema.NumberFromString.pipe(Schema.compose(Fence)), attempts: Schema.Int })), pending)
        for (const work of workRows) {
          const target = work.spec.target
          if (!target) return yield* new InfrastructureFailure({ operation: "reconcile", message: "Work target absent from plan" })
          yield* tx.query(`UPDATE lab_attempts SET ended_at=clock_timestamp(),detail=$4 WHERE run_id=$1 AND work_id=$2 AND fence=$3 AND ended_at IS NULL`,
            [run.run_id, work.work_id, work.fence, "Worker claim expired or run was cancelled"])
          if (!run.expired && run.state !== "Cancelling" && work.state === "Running" && work.attempts < 2) {
            const next = workFSM.transition(new Running({}), "Queued", {})
            yield* tx.query("UPDATE lab_work SET state=$3,fence=fence+1,worker=NULL,claim_expires_at=NULL WHERE run_id=$1 AND work_id=$2", [run.run_id, work.work_id, next._tag])
            yield* tx.query("INSERT INTO lab_events(run_id,kind,detail) VALUES($1,'infrastructure-retry',$2)", [run.run_id, work.work_id])
            continue
          }
          const now = new Date().toISOString()
          const result: WorkResult = { output: Option.none(), cleanupErrors: [], cases: target.cases.map(c => ({ targetId: target.target.id, caseId: c.id,
            harness: c.harness, startedAt: now, endedAt: now, evidence: [], outcome: {
              status: run.state === "Cancelling" ? "cancelled" : "blocked",
              detail: run.state === "Cancelling" ? "Run cancelled" : run.expired ? "Run deadline expired" : work.state === "Queued" ? "Producer failed to admit clean packages" : "Worker disconnected after the permitted infrastructure retry",
            } })) }
          const json = yield* Schema.encode(Schema.parseJson(WorkResult))(result).pipe(Effect.orDie)
          const current = work.state === "Running" ? new Running({}) : new Queued({})
          const next = workFSM.transition(current, "Finished", {})
          yield* tx.query("UPDATE lab_work SET state=$3,fence=fence+1,result=$4 WHERE run_id=$1 AND work_id=$2", [run.run_id, work.work_id, next._tag, json])
        }
        }
        const unfinished = yield* tx.query(`SELECT 1 FROM lab_work WHERE run_id=$1 AND state <> 'Finished'
          UNION ALL SELECT 1 FROM lab_leases WHERE run_id=$1 AND state <> 'Released' LIMIT 1`, [run.run_id])
        if (unfinished.length) continue
        const completed = yield* tx.query("SELECT spec,result FROM lab_work WHERE run_id=$1 ORDER BY work_id", [run.run_id])
        const results = yield* decodeRow(Schema.Array(Schema.Struct({ spec: Schema.parseJson(WorkSpec), result: Schema.parseJson(WorkResult) })), completed)
        const result = RunResult.make({ schemaVersion: 1, runId: run.run_id, plan: run.plan, startedAt: run.created_at.toISOString(), endedAt: new Date().toISOString(),
          cases: run.plan.targets.flatMap(target => {
            const consumer = results.find(row => row.spec.kind === "test" && row.spec.target.target.id === target.target.id)!
            const producerId = consumer.spec.kind === "test" ? Option.getOrUndefined(consumer.spec.producer) : undefined
            const producer = results.find(row => row.spec.id === producerId)
            return [...(producer?.result.cases ?? []).map(test => ({ ...test, targetId: target.target.id,
              outcome: { ...test.outcome, detail: `${test.outcome.detail} [producer ${producer!.spec.id}]` } })), ...consumer.result.cases]
          }), cleanupErrors: results.flatMap(r => r.result.cleanupErrors) })
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
