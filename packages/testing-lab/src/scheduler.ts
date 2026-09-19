import { Cause, Context, DateTime, Duration, Effect, Exit, Layer, Option, Schema } from "effect"
import { InfrastructureFailure, LeaseId, Provider } from "./domain"
import { Allocating, LeaseStore, type LeaseClaim, type LeaseRecord } from "./lease"
import { MachineAllocator, type Machine } from "./machines"
import { RunStore } from "./run-store"
import { WorkStore, type WorkAssignment, type TargetResult } from "./work-store"

export interface MachineProviders {
  readonly allocators: ReadonlyMap<typeof Provider.Type, MachineAllocator>
}
export const MachineProviders = Context.GenericTag<MachineProviders>("@magnitudedev/testing-lab/MachineProviders")
export interface WorkerRunner {
  /** Executes only this immutable assignment; the runner does not receive cloud credentials. */
  readonly run: (machine: Machine, assignment: WorkAssignment) => Effect.Effect<TargetResult, InfrastructureFailure>
}
export const WorkerRunner = Context.GenericTag<WorkerRunner>("@magnitudedev/testing-lab/WorkerRunner")
export const Reconciliation = Schema.Struct({ released: Schema.Int, errors: Schema.Array(Schema.String) })
export interface Scheduler {
  readonly next: (worker: string) => Effect.Effect<boolean, InfrastructureFailure>
  readonly reconcile: (janitor: string) => Effect.Effect<typeof Reconciliation.Type, InfrastructureFailure>
}
export const Scheduler = Context.GenericTag<Scheduler>("@magnitudedev/testing-lab/Scheduler")
const seconds = 60
const heartbeat = <A, E, R, E2, R2>(work: Effect.Effect<A, E, R>, beat: Effect.Effect<unknown, E2, R2>) =>
  Effect.raceFirst(work, Effect.forever(Effect.sleep("10 seconds").pipe(Effect.zipRight(beat))).pipe(Effect.interruptible))
const fail = (message: string) => new InfrastructureFailure({ operation: "scheduler", message })
const blocked = (assignment: WorkAssignment, detail: string): TargetResult => {
  const now = new Date().toISOString()
  return { cases: assignment.target.cases.map(c => ({ targetId: assignment.target.target.id, caseId: c.id, harness: c.harness,
    startedAt: now, endedAt: now, evidence: [], outcome: { status: "blocked", detail } })), cleanupErrors: [] }
}
const causeDetail = <E>(cause: Cause.Cause<E>) => {
  const error = Cause.failureOption(cause)
  return Option.isSome(error) && Schema.is(InfrastructureFailure)(error.value) ? error.value.message
    : Cause.isInterruptedOnly(cause) ? "Worker was interrupted" : "Worker execution failed; inspect scheduler diagnostics"
}
export const schedulerLayer = (cleanupTimeout: Duration.DurationInput = "20 minutes") => Layer.effect(Scheduler, Effect.gen(function* () {
  const providers = yield* MachineProviders
  const runner = yield* WorkerRunner
  const work = yield* WorkStore
  const leases = yield* LeaseStore
  const runs = yield* RunStore
  const release = (lease: LeaseRecord, allocator: MachineAllocator, janitor: string, known: readonly Machine[] = []) => Effect.gen(function* () {
    const cleanup = yield* leases.release(lease.state.leaseId, janitor, seconds)
    const claim: LeaseClaim = { leaseId: cleanup.state.leaseId, fence: cleanup.fence }
    const remove = Effect.gen(function* () {
      const inventory = yield* allocator.inventory()
      const owned = [...known, ...inventory].filter(m => m.tags.leaseId === lease.state.leaseId)
      for (const machine of owned) {
        if (machine.tags.runId !== lease.state.runId) return yield* fail("Provider ownership differs from durable lease")
        yield* allocator.release(machine)
      }
      if ((yield* allocator.inventory()).some(m => m.tags.leaseId === lease.state.leaseId)) return yield* fail("Provider allocation remains after cleanup")
      if (cleanup.state._tag !== "Released") yield* leases.released(claim)
    })
    if (cleanup.state._tag === "Released") yield* remove
    else yield* heartbeat(remove, leases.heartbeat(claim, seconds))
  }).pipe(Effect.mapError(error => fail(`Cleanup failed: ${error.message}`)))
  return {
    next: worker => Effect.gen(function* () {
      const assignment = yield* work.claim(worker, seconds)
      if (Option.isNone(assignment)) return false
      const job = assignment.value
      const target = job.target.target
      const allocator = providers.allocators.get(target.provider)
      const restricted = job.plan.request.trust === "untrusted-ci" && (target.provider === "spark" || target.provider === "local")
      if (job.target.blockers.length || !allocator || restricted) {
        yield* work.finish(job.claim, blocked(job, restricted ? "Untrusted CI cannot execute on local or office hardware" : job.target.blockers.join("; ") || `Provider ${target.provider} is not configured`)).pipe(Effect.mapError(e => fail(e.message)))
        yield* work.reconcile()
        return true
      }
      const run = Effect.gen(function* () {
        const allocation = new Allocating({ leaseId: LeaseId.make(`lease-${crypto.randomUUID()}`), runId: job.claim.runId,
          targetId: target.id, provider: target.provider, resourceName: `ml-${crypto.randomUUID().replaceAll("-", "").slice(0, 12)}`, expiresAt: job.deadline })
        const lease = yield* leases.reserve(allocation, worker, seconds).pipe(Effect.mapError(e => fail(e.message)))
        const claim: LeaseClaim = { leaseId: lease.state.leaseId, fence: lease.fence }
        let machine: Machine | undefined
        // Cleanup executes on success, failure, defect and cancellation. Its own finite timeout
        // leaves a discoverable Releasing lease for the independent janitor when the provider stalls.
        const exit = yield* Effect.uninterruptibleMask(restore => Effect.gen(function* () {
          const outcome = yield* restore(heartbeat(Effect.gen(function* () {
            machine = yield* allocator.ensure(allocation, target)
            yield* leases.ready(claim)
            return yield* runner.run(machine, job)
          }), leases.heartbeat(claim, seconds))).pipe(Effect.exit)
          const cleaned = yield* release(lease, allocator, `${worker}-cleanup`, machine ? [machine] : []).pipe(
            Effect.interruptible, Effect.timeoutFail({ duration: cleanupTimeout, onTimeout: () => fail("Cleanup timed out; janitor retains responsibility") }), Effect.exit)
          return { outcome, cleaned }
        }))
        const result = Exit.isSuccess(exit.outcome) ? exit.outcome.value : blocked(job, causeDetail(exit.outcome.cause))
        const cleanupErrors = Exit.isFailure(exit.cleaned) ? [...result.cleanupErrors, causeDetail(exit.cleaned.cause)] : result.cleanupErrors
        yield* work.finish(job.claim, { ...result, cleanupErrors }).pipe(Effect.mapError(e => fail(e.message)))
      })
      yield* heartbeat(run, work.heartbeat(job.claim, seconds)).pipe(Effect.mapError(e => fail(e.message)))
      yield* work.reconcile()
      return true
    }),
    reconcile: janitor => Effect.gen(function* () {
      const records = yield* leases.list()
      let released = 0
      const errors: string[] = []
      for (const [provider, allocator] of providers.allocators) {
        const observed = yield* allocator.inventory().pipe(Effect.either)
        if (observed._tag === "Left") { errors.push(`${provider}: ${observed.left.message}`); continue }
        const inventory = observed.right
        for (const lease of records.filter(l => l.state.provider === provider)) {
          const machines = inventory.filter(m => m.tags.leaseId === lease.state.leaseId)
          if (lease.state._tag === "Released" && machines.length === 0) continue
          const run = yield* runs.get(lease.state.runId).pipe(Effect.either)
          if (run._tag === "Left" && run.left._tag !== "RunNotFound") { errors.push(run.left.message); continue }
          const now = Date.now()
          const expired = DateTime.toEpochMillis(lease.state.expiresAt) <= now || DateTime.toEpochMillis(lease.claimExpiresAt) <= now
          if (lease.state._tag === "Releasing" && DateTime.toEpochMillis(lease.claimExpiresAt) > now && lease.worker !== janitor) continue
          const stopped = run._tag === "Left" || run.right.state._tag === "Cancelling" || run.right.state._tag === "Finished"
          if (!expired && !stopped && lease.state._tag !== "Releasing" && lease.state._tag !== "Released") continue
          const result = yield* release(lease, allocator, janitor, machines).pipe(Effect.either)
          if (result._tag === "Left") errors.push(result.left.message)
          else released++
        }
        // The database may have been restored after an allocation. Absolute provider expiry
        // bounds these orphan costs without authorizing deletion of untagged resources.
        const ids = new Set(records.map(l => l.state.leaseId))
        for (const machine of inventory.filter(m => !ids.has(m.tags.leaseId) && DateTime.toEpochMillis(m.tags.expiresAt) <= Date.now())) {
          const result = yield* allocator.release(machine).pipe(Effect.either)
          if (result._tag === "Left") errors.push(result.left.message)
          else released++
        }
      }
      yield* work.reconcile()
      return { released, errors }
    }),
  } satisfies Scheduler
}))

export const SchedulerLive = schedulerLayer()
