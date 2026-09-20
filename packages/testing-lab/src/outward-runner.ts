import { assignmentInputs } from "./work-store"
import { Context, DateTime, Effect, Layer, Option, Redacted, Schema } from "effect"
import { posix, win32 } from "node:path"
import { InfrastructureFailure, Provider } from "./domain"
import { InputRegistry } from "./inputs"
import { Machine } from "./machines"
import { WorkerRunner } from "./scheduler"
import { validateTargetResult } from "./work-store"
import { WorkerInvocation } from "./worker-protocol"
import { WorkerResults } from "./worker-results"
import { WorkerTickets } from "./worker-tickets"
import { GuestRuntime } from "./worker-runner"
import { ArtifactStore } from "./artifact-store"
import { retainWorkerDiagnostic } from "./worker-diagnostics"

export const WorkerLaunch = Schema.Struct({ executable: Schema.NonEmptyString, args: Schema.Array(Schema.String), root: Schema.NonEmptyString,
  origin: Schema.NonEmptyString, token: Schema.Redacted(Schema.NonEmptyString), deadline: Schema.DateTimeUtc })
export const WorkerExit = Schema.Struct({ state: Schema.Literal("Succeeded", "Failed", "TimedOut", "Canceled"),
  code: Schema.optionalWith(Schema.Int, { as: "Option", exact: true }),
  output: Schema.optionalWith(Schema.Redacted(Schema.String), { as: "Option", exact: true }).pipe(Schema.withConstructorDefault(() => Option.none())) })
export interface WorkerBootstrap {
  /** Start once and return after delivery. Provider credentials stay in this privileged adapter. */
  readonly poll: (machine: Machine) => Effect.Effect<Option.Option<typeof WorkerExit.Type>, InfrastructureFailure>
  readonly start: (machine: Machine, launch: typeof WorkerLaunch.Type) => Effect.Effect<void, InfrastructureFailure>
}
export interface WorkerBootstraps { readonly providers: ReadonlyMap<typeof Provider.Type, WorkerBootstrap> }
export const WorkerBootstraps = Context.GenericTag<WorkerBootstraps>("@magnitudedev/testing-lab/WorkerBootstraps")
export const OutwardRunnerConfig = Schema.Struct({ origin: Schema.NonEmptyString, runtimes: Schema.Array(GuestRuntime), pollMs: Schema.Int.pipe(Schema.between(10, 30_000)) })
const fail = (message: string) => new InfrastructureFailure({ operation: "outward-runner", message })

export const outwardWorkerRunner = (config: typeof OutwardRunnerConfig.Type) => Layer.effect(WorkerRunner, Effect.gen(function* () {
  const origin = yield* Effect.try({ try: () => new URL(config.origin), catch: () => fail("Invalid guest coordinator origin") })
  if (origin.username || origin.password || origin.pathname !== "/" || origin.search || origin.hash ||
    (origin.protocol !== "https:" && !(origin.protocol === "http:" && ["localhost", "127.0.0.1", "[::1]"].includes(origin.hostname)))) return yield* fail("Guest coordinator requires an HTTPS origin or loopback HTTP")
  const tickets = yield* WorkerTickets
  const results = yield* WorkerResults
  const inputs = yield* InputRegistry
  const bootstraps = yield* WorkerBootstraps
  const objects = yield* ArtifactStore
  return {
    run: (machine, assignment) => Effect.gen(function* () {
      const matches = config.runtimes.filter(runtime => runtime.provider === machine.provider && runtime.artifactHost === assignment.target.target.artifactHost)
      const bootstrap = bootstraps.providers.get(machine.provider)
      if (matches.length !== 1 || !bootstrap) return yield* fail("Exactly one guest runtime and bootstrap must match the worker")
      const runtime = matches[0]!
      if (runtime.disposable && (machine.provider === "local" || machine.provider === "spark")) return yield* fail("Shared hosts cannot grant a disposable OS-user context")
      if (machine.tags.runId !== assignment.claim.runId || assignment.claim.targetId !== assignment.target.target.id) return yield* fail("Worker ownership differs from assignment")
      if (assignment.plan.request.trust === "untrusted-ci" && (!runtime.disposable || machine.provider === "local" || machine.provider === "spark")) return yield* fail("Untrusted work requires disposable cloud execution")
      const deadline = Math.min(DateTime.toEpochMillis(machine.tags.expiresAt), DateTime.toEpochMillis(assignment.deadline))
      if (deadline <= Date.now()) return yield* fail("Worker allocation has expired")
      const path = assignment.target.target.os === "windows" ? win32 : posix
      const base = machine.provider === "local" ? machine.root : runtime.root
      if (!path.isAbsolute(base)) return yield* fail("Guest root must be absolute")
      const root = path.join(base, machine.tags.leaseId, `attempt-${assignment.claim.fence}`)
      yield* Effect.forEach(assignmentInputs(assignment), input => inputs.require(assignment.plan.request.owner, input), { discard: true })
      const cleanupErrors: string[] = []
      const result = yield* Effect.scoped(Effect.gen(function* () {
        const invocation = WorkerInvocation.make({ schemaVersion: 1, assignment, disposable: runtime.disposable, port: runtime.port, model: runtime.model })
        const ticket = yield* Effect.acquireRelease(tickets.issue(invocation), ticket => tickets.revoke(ticket.id).pipe(
          Effect.interruptible, Effect.timeoutFail({ duration: "30 seconds", onTimeout: () => fail("Worker credential revocation timed out") }),
          Effect.catchAll(error => Effect.sync(() => { cleanupErrors.push(`Worker credential: ${error.message}`) }))))
        yield* bootstrap.start(machine, WorkerLaunch.make({ executable: runtime.executable, args: runtime.args, root, origin: origin.origin, token: ticket.token, deadline: DateTime.unsafeMake(deadline) }))
        let checkedAt = 0
        for (;;) {
          yield* tickets.authorize(ticket.token)
          const reply = yield* results.read(assignment.claim)
          if (Option.isSome(reply)) {
            yield* validateTargetResult(assignment.target, reply.value.result)
            return reply.value.result
          }
          if (Date.now() - checkedAt >= 10_000) {
            const exited = yield* bootstrap.poll(machine)
            checkedAt = Date.now()
            if (Option.isSome(exited)) {
              // Receipt may commit between the first database read and provider observation.
              const delivered = yield* results.read(assignment.claim)
              if (Option.isSome(delivered)) {
                yield* validateTargetResult(assignment.target, delivered.value.result)
                return delivered.value.result
              }
              const diagnostic = yield* retainWorkerDiagnostic(machine, "execution", Option.match(exited.value.output, {
                onNone: () => "Provider did not return execution output", onSome: Redacted.value,
              }), [Redacted.value(ticket.token)]).pipe(Effect.provideService(ArtifactStore, objects), Effect.either)
              return yield* new InfrastructureFailure({ operation: "outward-runner",
                message: `Guest process ended ${exited.value.state}${Option.match(exited.value.code, { onNone: () => "", onSome: code => ` (exit ${code})` })} without an accepted result; ${diagnostic._tag === "Right" ? "execution diagnostics retained in run evidence" : "could not retain execution diagnostics"}`,
                evidence: diagnostic._tag === "Right" ? Option.some([diagnostic.right]) : Option.none() })
            }
          }
          yield* Effect.sleep(config.pollMs)
        }
      }).pipe(Effect.timeoutFail({ duration: Math.max(1, deadline - Date.now()), onTimeout: () => fail("Guest execution exceeded its allocation deadline") }))).pipe(
        Effect.mapError(error => new InfrastructureFailure({ operation: "outward-runner",
          message: `${error._tag === "WorkerAccessDenied" ? "Worker authority ended before result collection" : error.message}${cleanupErrors.length ? `; ${cleanupErrors.join("; ")}` : ""}`,
          evidence: error._tag === "InfrastructureFailure" ? error.evidence : Option.none() })))
      return { ...result, cleanupErrors: [...result.cleanupErrors, ...cleanupErrors] }
    }).pipe(Effect.mapError(error => error._tag === "InfrastructureFailure" ? error : fail(error.message))),
  } satisfies WorkerRunner
}))
