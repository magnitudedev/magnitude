import { describe, expect, it } from "vitest"
import { Deferred, Effect, Option, Ref, Schema } from "effect"
import {
  Address,
  AddressMatch,
  AddressTarget,
  EventEngine,
  Projection,
  Worker,
} from "../src"

type WorkEvent =
  | { readonly type: "lane_opened"; readonly laneId: string }
  | { readonly type: "value_set"; readonly laneId: string; readonly value: number }
  | { readonly type: "work_requested"; readonly laneId: string; readonly block: boolean }
  | { readonly type: "lane_closed"; readonly laneId: string }

const Lanes = Address.space({ name: "lane", key: Schema.String })
const WorkEngine = EventEngine.define<WorkEvent>({ discriminator: "type" })
const byLane = (event: WorkEvent) => AddressTarget.one(Lanes.at(event.laneId))

const Values = WorkEngine.Projection.partitioned({
  name: "Values",
  by: Lanes,
  state: Schema.Number,
  initial: () => 0,
  handlers: (on) => [
    on.event("value_set", { route: byLane }, ({ event }) => Projection.set(event.value)),
  ],
})

describe("EventEngine workers", () => {
  it("does not run workers before activation and observes settled projection truth", async () => {
    const observed = await Effect.runPromise(Ref.make<readonly number[]>([]))
    const Observer = WorkEngine.Worker.define({
      name: "Observer",
      reads: [Values],
      handlers: (on) => [
        on.event("value_set", ({ event, read }) => Effect.gen(function* () {
          const value = yield* read.at(Values, Lanes.at(event.laneId))
          yield* Ref.update(observed, (values) => [...values, Option.getOrElse(value, () => -1)])
        })),
      ],
    })
    const definition = WorkEngine.make({ name: "observer", projections: [Values], workers: [Observer] })

    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const runtime = yield* definition.build
      yield* runtime.accept({ type: "value_set", laneId: "a", value: 1 })
      yield* runtime.activate()
      yield* runtime.accept({ type: "value_set", laneId: "a", value: 2 })
      yield* runtime.awaitIdle
      expect(yield* Ref.get(observed)).toEqual([2])
    })))
  })

  it("admits lanes independently, and interruption reaches queued work, which still runs, and permits new work", async () => {
    const started = await Effect.runPromise(Deferred.make<void>())
    const interrupted = await Effect.runPromise(Deferred.make<void>())
    const completed = await Effect.runPromise(Ref.make<readonly string[]>([]))

    const Runner = WorkEngine.Worker.define({
      name: "Runner",
      partition: {
        by: Lanes,
        admission: (on) => [
          on.event("lane_opened", { route: byLane }, Worker.admit),
          on.event("lane_closed", { route: byLane }, Worker.close),
        ],
      },
      handlers: (on) => [
        on.event("work_requested", { route: byLane }, ({ event, address, interruption }) =>
          event.block
            ? Deferred.succeed(started, undefined).pipe(
                Effect.zipRight(Effect.never),
                Effect.onInterrupt(() => Deferred.succeed(interrupted, undefined)),
              )
            : Effect.flatMap(interruption, (observed) => Ref.update(completed, (values) => [
                ...values,
                Option.match(observed, { onNone: () => address.key, onSome: ({ _tag }) => `${address.key}:${_tag}` }),
              ]))),
      ],
    })
    const definition = WorkEngine.make({ name: "runner", projections: [], workers: [Runner] })

    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const runtime = yield* definition.build
      yield* runtime.activate()
      yield* runtime.accept({ type: "work_requested", laneId: "a", block: false })
      yield* runtime.awaitIdle
      expect(yield* Ref.get(completed)).toEqual([])

      yield* runtime.accept({ type: "lane_opened", laneId: "a" })
      yield* runtime.accept({ type: "work_requested", laneId: "a", block: true })
      yield* Deferred.await(started)
      yield* runtime.accept({ type: "work_requested", laneId: "a", block: false })
      yield* runtime.interrupt(Lanes.at("a"))
      yield* Deferred.await(interrupted)

      yield* runtime.accept({ type: "work_requested", laneId: "a", block: false })
      yield* runtime.awaitIdle
      expect(yield* Ref.get(completed)).toEqual(["a:Interrupted", "a"])

      yield* runtime.accept({ type: "lane_closed", laneId: "a" })
      yield* runtime.accept({ type: "work_requested", laneId: "a", block: false })
      yield* runtime.awaitIdle
      expect(yield* Ref.get(completed)).toEqual(["a:Interrupted", "a"])
    })))
  })

  it("runs different address lanes independently", async () => {
    const firstStarted = await Effect.runPromise(Deferred.make<void>())
    const releaseFirst = await Effect.runPromise(Deferred.make<void>())
    const secondCompleted = await Effect.runPromise(Deferred.make<void>())
    const ParallelRunner = WorkEngine.Worker.define({
      name: "ParallelRunner",
      partition: {
        by: Lanes,
        admission: (on) => [on.event("lane_opened", { route: byLane }, Worker.admit)],
      },
      handlers: (on) => [
        on.event("work_requested", { route: byLane }, ({ address }) =>
          address.key === "a"
            ? Deferred.succeed(firstStarted, undefined).pipe(Effect.zipRight(Deferred.await(releaseFirst)))
            : Deferred.succeed(secondCompleted, undefined)),
      ],
    })
    const definition = WorkEngine.make({ name: "parallel", projections: [], workers: [ParallelRunner] })

    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const runtime = yield* definition.build
      yield* runtime.activate()
      yield* runtime.accept({ type: "lane_opened", laneId: "a" })
      yield* runtime.accept({ type: "lane_opened", laneId: "b" })
      yield* runtime.accept({ type: "work_requested", laneId: "a", block: true })
      yield* Deferred.await(firstStarted)
      yield* runtime.accept({ type: "work_requested", laneId: "b", block: false })
      yield* Deferred.await(secondCompleted)
      yield* Deferred.succeed(releaseFirst, undefined)
      yield* runtime.awaitIdle
    })))
  })

  it("distinguishes exact interruption from explicit subtree interruption", async () => {
    type ProcessEvent = { readonly type: "run"; readonly forkId: string; readonly pid: number }
    const Forks = Address.space({ name: "fork", key: Schema.String })
    const Processes = Address.space({ name: "process", parent: Forks, key: Schema.Number })
    const ProcessEngine = EventEngine.define<ProcessEvent>({ discriminator: "type" })
    const process = Processes.at(Forks.at("feature"), 1)
    const started = await Effect.runPromise(Deferred.make<void>())
    const interrupted = await Effect.runPromise(Deferred.make<void>())
    const ProcessRunner = ProcessEngine.Worker.define({
      name: "ProcessRunner",
      partition: { by: Processes, initial: [process], admission: () => [] },
      handlers: (on) => [
        on.event("run", {
          route: () => AddressTarget.one(process),
        }, () => Deferred.succeed(started, undefined).pipe(
          Effect.zipRight(Effect.never),
          Effect.onInterrupt(() => Deferred.succeed(interrupted, undefined)),
        )),
      ],
    })
    const definition = ProcessEngine.make({ name: "processes", projections: [], workers: [ProcessRunner] })

    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const runtime = yield* definition.build
      yield* runtime.activate()
      yield* runtime.accept({ type: "run", forkId: "feature", pid: 1 })
      yield* Deferred.await(started)

      yield* runtime.interrupt(Forks.at("feature"))
      expect(Option.isNone(yield* Deferred.poll(interrupted))).toBe(true)

      yield* runtime.interrupt(AddressMatch.subtree(Forks.at("feature")))
      yield* Deferred.await(interrupted)
      yield* runtime.awaitIdle
    })))
  })
})
