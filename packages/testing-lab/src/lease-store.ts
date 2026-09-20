import { DateTime, Effect, Layer, Schema } from "effect"
import { Database, decodeRow, type DatabaseSession } from "./database"
import { Allocating, Fence, LeaseConflict, LeaseRecord, LeaseStore, Ready, Released, Releasing, StaleLease, leaseFSM, type LeaseClaim } from "./lease"
import { InfrastructureFailure, LeaseId, Provider, RunId, TargetId } from "./domain"

import { WorkId } from "./work-identity"

const Row = Schema.Struct({ lease_id: LeaseId, run_id: RunId, work_id: WorkId, work_fence: Schema.NumberFromString.pipe(Schema.compose(Fence)), target_id: TargetId, provider: Provider,
  resource_name: Schema.String, state: Schema.Literal("Allocating", "Ready", "Releasing", "Released"),
  expires_at: Schema.DateFromSelf, fence: Schema.NumberFromString.pipe(Schema.compose(Fence)),
  worker: Schema.String, claim_expires_at: Schema.DateFromSelf })
const record = (value: unknown) => decodeRow(Row, value).pipe(Effect.map(row => {
  const props = { leaseId: row.lease_id, runId: row.run_id, workId: row.work_id, workFence: row.work_fence, targetId: row.target_id, provider: row.provider,
    resourceName: row.resource_name, expiresAt: DateTime.unsafeMake(row.expires_at) }
  const constructors = { Allocating, Ready, Releasing, Released }
  return LeaseRecord.make({ state: new constructors[row.state](props), fence: row.fence, worker: row.worker,
    claimExpiresAt: DateTime.unsafeMake(row.claim_expires_at) })
}))
const validSeconds = (seconds: number) => Number.isInteger(seconds) && seconds >= 1 && seconds <= 3600
const timingError = () => new InfrastructureFailure({ operation: "lease", message: "Claim duration must be 1–3600 seconds" })
const owned = (db: DatabaseSession, claim: LeaseClaim) => Effect.gen(function* () {
  const rows = yield* db.query(`SELECT * FROM lab_leases WHERE lease_id=$1 AND fence=$2
    AND claim_expires_at > clock_timestamp()
    AND (state='Releasing' OR expires_at > clock_timestamp()) FOR UPDATE`, [claim.leaseId, claim.fence])
  if (!rows[0]) return yield* new StaleLease({ leaseId: claim.leaseId })
  return yield* record(rows[0])
})
const persistState = (db: DatabaseSession, value: LeaseRecord) => db.query(
  "UPDATE lab_leases SET state=$2 WHERE lease_id=$1 RETURNING *", [value.state.leaseId, value.state._tag],
).pipe(Effect.flatMap(rows => record(rows[0])))

export const LeaseStoreLive = Layer.effect(LeaseStore, Effect.gen(function* () {
  const db = yield* Database
  return {
    reserve: (state, worker, seconds) => !validSeconds(seconds) ? Effect.fail(timingError()) : db.transaction(tx => Effect.gen(function* () {
      // Serialize reservations so uniqueness conflicts are represented as domain failures.
      yield* tx.query("SELECT pg_advisory_xact_lock(91826002)")
      const existing = yield* tx.query(`SELECT * FROM lab_leases WHERE lease_id=$1 OR (provider=$2 AND resource_name=$3)
        OR ($2='spark' AND provider='spark' AND state <> 'Released')`, [state.leaseId, state.provider, state.resourceName])
      if (existing.length) return yield* new LeaseConflict({ message: "Lease, resource name or Spark reservation already exists" })
      const time = yield* tx.query("SELECT $1::timestamptz > clock_timestamp() AS valid", [DateTime.toDate(state.expiresAt)])
      if (time[0]?.valid !== true) return yield* new LeaseConflict({ message: "Cannot reserve an expired lease" })
      const rows = yield* tx.query(`INSERT INTO lab_leases
        (lease_id,run_id,target_id,provider,resource_name,state,expires_at,worker,claim_expires_at,work_id,work_fence)
        VALUES ($1,$2,$3,$4,$5,'Allocating',$6,$7,LEAST($6,clock_timestamp()+$8*interval '1 second'),$9,$10) RETURNING *`,
      [state.leaseId, state.runId, state.targetId, state.provider, state.resourceName, DateTime.toDate(state.expiresAt), worker, seconds, state.workId, state.workFence])
      return yield* record(rows[0])
    })),
    ready: claim => db.transaction(tx => Effect.gen(function* () {
      const lease = yield* owned(tx, claim)
      if (lease.state._tag !== "Allocating") return yield* new StaleLease({ leaseId: claim.leaseId })
      return yield* persistState(tx, { ...lease, state: leaseFSM.transition(lease.state, "Ready", {}) })
    })),
    heartbeat: (claim, seconds) => !validSeconds(seconds) ? Effect.fail(timingError()) : db.transaction(tx => Effect.gen(function* () {
      const rows = yield* tx.query(`UPDATE lab_leases SET claim_expires_at=CASE WHEN state='Releasing'
        THEN clock_timestamp()+$3*interval '1 second' ELSE LEAST(expires_at,clock_timestamp()+$3*interval '1 second') END
        WHERE lease_id=$1 AND fence=$2 AND state IN ('Allocating','Ready','Releasing')
        AND claim_expires_at > clock_timestamp() AND (state='Releasing' OR expires_at > clock_timestamp()) RETURNING *`, [claim.leaseId, claim.fence, seconds])
      if (!rows[0]) return yield* new StaleLease({ leaseId: claim.leaseId })
      return yield* record(rows[0])
    })),
    release: (leaseId, worker, seconds) => !validSeconds(seconds) ? Effect.fail(timingError()) : db.transaction(tx => Effect.gen(function* () {
      const rows = yield* tx.query("SELECT * FROM lab_leases WHERE lease_id=$1 FOR UPDATE", [leaseId])
      if (!rows[0]) return yield* new LeaseConflict({ message: "Cannot release an unknown lease" })
      const lease = yield* record(rows[0])
      if (lease.state._tag === "Released") return lease
      if (lease.state._tag === "Releasing") {
        const active = yield* tx.query("SELECT claim_expires_at > clock_timestamp() AS active FROM lab_leases WHERE lease_id=$1", [leaseId])
        if (active[0]?.active === true) {
          if (lease.worker === worker) return lease
          return yield* new LeaseConflict({ message: "Another janitor owns the active cleanup claim" })
        }
      }
      const state = lease.state._tag === "Releasing" ? lease.state : leaseFSM.transition(lease.state, "Releasing", {})
      const changed = yield* tx.query(`UPDATE lab_leases SET state=$2, fence=fence+1, worker=$3,
        claim_expires_at=clock_timestamp()+$4*interval '1 second' WHERE lease_id=$1 RETURNING *`, [leaseId, state._tag, worker, seconds])
      return yield* record(changed[0])
    })),
    released: claim => db.transaction(tx => Effect.gen(function* () {
      const lease = yield* owned(tx, claim)
      if (lease.state._tag !== "Releasing") return yield* new StaleLease({ leaseId: claim.leaseId })
      return yield* persistState(tx, { ...lease, state: leaseFSM.transition(lease.state, "Released", {}) })
    })),
    list: () => db.query("SELECT * FROM lab_leases ORDER BY lease_id").pipe(Effect.flatMap(rows => Effect.forEach(rows, record))),
  } satisfies LeaseStore
}))
