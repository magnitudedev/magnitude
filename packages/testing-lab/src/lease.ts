import { Context, Effect, Schema } from "effect"
import { defineFSM } from "@magnitudedev/utils/fsm"
import { InfrastructureFailure, LeaseId, Provider, RunId, TargetId } from "./domain"

export const Fence = Schema.Int.pipe(Schema.positive(), Schema.brand("LeaseFence"))
export type Fence = typeof Fence.Type
export const LeaseClaim = Schema.Struct({ leaseId: LeaseId, fence: Fence })
export type LeaseClaim = typeof LeaseClaim.Type
const identity = { leaseId: LeaseId, runId: RunId, targetId: TargetId, provider: Provider,
  resourceName: Schema.NonEmptyString, expiresAt: Schema.DateTimeUtc }
export class Allocating extends Schema.TaggedClass<Allocating>()("Allocating", identity) {}
export class Ready extends Schema.TaggedClass<Ready>()("Ready", identity) {}
export class Releasing extends Schema.TaggedClass<Releasing>()("Releasing", identity) {}
export class Released extends Schema.TaggedClass<Released>()("Released", identity) {}
export const LeaseState = Schema.Union(Allocating, Ready, Releasing, Released)
export type LeaseState = typeof LeaseState.Type
export const leaseFSM = defineFSM({ Allocating, Ready, Releasing, Released }, {
  Allocating: ["Ready", "Releasing"], Ready: ["Releasing"], Releasing: ["Released"], Released: [],
})
export const LeaseRecord = Schema.Struct({ state: LeaseState, fence: Fence,
  worker: Schema.NonEmptyString, claimExpiresAt: Schema.DateTimeUtc })
export type LeaseRecord = typeof LeaseRecord.Type
export class StaleLease extends Schema.TaggedError<StaleLease>()("StaleLease", { leaseId: LeaseId }) {}
export class LeaseConflict extends Schema.TaggedError<LeaseConflict>()("LeaseConflict", { message: Schema.String }) {}
export interface LeaseStore {
  /** Persist deterministic identity before contacting a provider. */
  readonly reserve: (state: Allocating, worker: string, claimSeconds: number) => Effect.Effect<LeaseRecord, InfrastructureFailure | LeaseConflict>
  readonly ready: (claim: LeaseClaim) => Effect.Effect<LeaseRecord, InfrastructureFailure | StaleLease>
  readonly heartbeat: (claim: LeaseClaim, claimSeconds: number) => Effect.Effect<LeaseRecord, InfrastructureFailure | StaleLease>
  /** Revokes worker authority before invoking deletion. Safe for the janitor to retry. */
  readonly release: (leaseId: LeaseId, worker: string, claimSeconds: number) => Effect.Effect<LeaseRecord, InfrastructureFailure | LeaseConflict>
  readonly released: (claim: LeaseClaim) => Effect.Effect<LeaseRecord, InfrastructureFailure | StaleLease>
  readonly list: () => Effect.Effect<ReadonlyArray<LeaseRecord>, InfrastructureFailure>
}
export const LeaseStore = Context.GenericTag<LeaseStore>("@magnitudedev/testing-lab/LeaseStore")
