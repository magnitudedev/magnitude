import { HttpRouter, HttpServer } from "@effect/platform"
import { Context, Effect, Layer, Schema } from "effect"
import { api, Authenticator } from "./api"
import { initializeDatabase } from "./database"
import { InputRegistryLive } from "./inputs"
import { LeaseStoreLive } from "./lease-store"
import { runStoreLayer } from "./run-store"
import { Scheduler, SchedulerLive } from "./scheduler"
import { WorkStoreLive } from "./work-store"
import { WorkerTicketsLive } from "./worker-tickets"
import { workerApi } from "./worker-api"
import { WorkerInputsLive } from "./worker-inputs"

export const CoordinatorConfig = Schema.Struct({ instance: Schema.NonEmptyString,
  concurrency: Schema.Int.pipe(Schema.between(1, 64)), accountBudgetUsd: Schema.Number.pipe(Schema.positive(), Schema.finite()),
  pollMs: Schema.Int.pipe(Schema.between(1, 60_000)), reconcileMs: Schema.Int.pipe(Schema.between(1, 60_000)) })
/** Startup migrations and service instances are shared by HTTP handlers and supervised loops. */
export const startCoordinator = (config: typeof CoordinatorConfig.Type) => Effect.gen(function* () {
  yield* initializeDatabase
  const workerInputs = WorkerInputsLive.pipe(Layer.provide(Layer.merge(InputRegistryLive, WorkerTicketsLive)))
  const services = yield* Layer.build(Layer.mergeAll(InputRegistryLive, LeaseStoreLive, WorkStoreLive, WorkerTicketsLive, workerInputs, runStoreLayer(config.accountBudgetUsd)))
  const scheduler = Context.get(yield* Layer.build(SchedulerLive.pipe(Layer.provide(Layer.succeedContext(services)))), Scheduler)
  const auth = yield* Authenticator
  const server = yield* HttpServer.HttpServer
  yield* server.serve(HttpRouter.concat(api, workerApi).pipe(Effect.provide(Context.add(services, Authenticator, auth))))
  const workers = Array.from({ length: config.concurrency }, (_, index) => Effect.forever(
    scheduler.next(`${config.instance}-worker-${index}`).pipe(
      Effect.catchAll(error => Effect.logError(`Scheduler: ${error.message}`).pipe(Effect.as(false))),
      Effect.flatMap(worked => worked ? Effect.yieldNow() : Effect.sleep(config.pollMs)),
    )))
  const reconciler = Effect.forever(Effect.gen(function* () {
    const result = yield* scheduler.reconcile(`${config.instance}-janitor`).pipe(Effect.either)
    if (result._tag === "Left") yield* Effect.logError(`Reconciler: ${result.left.message}`)
    else for (const error of result.right.errors) yield* Effect.logError(`Reconciler: ${error}`)
    yield* Effect.sleep(config.reconcileMs)
  }))
  // A defect in any loop terminates the supervised service; typed infrastructure failures are
  // observed and retried after bounded polling. Scope closure interrupts work and performs cleanup.
  return { address: server.address, run: Effect.all([...workers, reconciler], { concurrency: "unbounded", discard: true }) }
})
