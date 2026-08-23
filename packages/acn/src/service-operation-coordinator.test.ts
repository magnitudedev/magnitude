import {
  Deferred,
  Effect,
  Exit,
  Fiber,
  Option,
  Ref,
} from "effect"
import { describe, expect, it } from "vitest"
import {
  makeServiceOperationCoordinator,
  type ServiceOperationDefinition,
} from "./service-operation-coordinator"

describe("ServiceOperationCoordinator", () => {
  it("owns one operation while equivalent callers join and conflicting callers observe it", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const coordinator = yield* makeServiceOperationCoordinator<string, string>(
        (left, right) => left === right,
      )
      const release = yield* Deferred.make<void>()
      const runs = yield* Ref.make(0)
      const terminalized = yield* Ref.make(0)
      const definition: ServiceOperationDefinition<string> = {
        commit: Effect.void,
        operation: Ref.update(runs, (count) => count + 1).pipe(
          Effect.zipRight(Deferred.await(release)),
        ),
        terminalize: () => Ref.update(terminalized, (count) => count + 1),
      }
      const request = (key: string) => Effect.succeed({
        key,
        whenIdle: Effect.succeed(Option.some(definition)),
      })

      const first = yield* coordinator.admit(request("same"))
      const joined = yield* coordinator.admit(request("same"))
      const conflicting = yield* coordinator.admit(request("different"))
      expect(first._tag).toBe("Current")
      expect(joined._tag).toBe("Current")
      expect(conflicting._tag).toBe("Conflicting")

      const interruptedWaiter = joined._tag === "Current"
        ? yield* joined.outcome.pipe(Effect.fork)
        : yield* Effect.die("expected current operation")
      yield* Fiber.interrupt(interruptedWaiter)
      expect(yield* Ref.get(runs)).toBe(1)

      yield* Deferred.succeed(release, undefined)
      if (first._tag === "Current") expect(Exit.isSuccess(yield* first.outcome)).toBe(true)
      expect(yield* Ref.get(terminalized)).toBe(1)
    })))
  })

  it("keeps preparation interruptible until the masked admission commit", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const coordinator = yield* makeServiceOperationCoordinator<string, never>(
        (left, right) => left === right,
      )
      const entered = yield* Deferred.make<void>()
      const releasePreparation = yield* Deferred.make<void>()
      const commits = yield* Ref.make(0)
      const runs = yield* Ref.make(0)

      const caller = yield* coordinator.admit(Effect.succeed({
        key: "operation",
        whenIdle: Deferred.succeed(entered, undefined).pipe(
          Effect.zipRight(Deferred.await(releasePreparation)),
          Effect.as(Option.some({
            commit: Ref.update(commits, (count) => count + 1),
            operation: Ref.update(runs, (count) => count + 1),
            terminalize: () => Effect.void,
          })),
        ),
      })).pipe(Effect.fork)

      yield* Deferred.await(entered)
      yield* Fiber.interrupt(caller)
      yield* Deferred.succeed(releasePreparation, undefined)
      yield* Effect.yieldNow()
      expect(yield* Ref.get(commits)).toBe(0)
      expect(yield* Ref.get(runs)).toBe(0)
    })))
  })

  it("shares typed failure and terminalizes defects before admitting the next operation", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const coordinator = yield* makeServiceOperationCoordinator<string, string>(
        (left, right) => left === right,
      )
      const exits = yield* Ref.make<readonly Exit.Exit<void, string>[]>([])
      const request = (key: string, operation: Effect.Effect<void, string>) =>
        Effect.succeed({
          key,
          whenIdle: Effect.succeed(Option.some({
            commit: Effect.void,
            operation,
            terminalize: (exit: Exit.Exit<void, string>) =>
              Ref.update(exits, (values) => [...values, exit]),
          })),
        })

      const failed = yield* coordinator.admit(request("failed", Effect.fail("expected")))
      if (failed._tag !== "Current") return yield* Effect.die("expected current operation")
      const failedExit = yield* failed.outcome
      expect(Exit.isFailure(failedExit)).toBe(true)

      const defect = yield* coordinator.admit(request("defect", Effect.die("unexpected")))
      if (defect._tag !== "Current") return yield* Effect.die("expected current operation")
      const defectExit = yield* defect.outcome
      expect(Exit.isFailure(defectExit)).toBe(true)
      expect((yield* Ref.get(exits)).length).toBe(2)

      const satisfied = yield* coordinator.admit(Effect.succeed({
        key: "next",
        whenIdle: Effect.succeed(Option.none()),
      }))
      expect(satisfied._tag).toBe("Satisfied")
    })))
  })

  it("releases ownership when the admission commit defects", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const coordinator = yield* makeServiceOperationCoordinator<string, never>(
        (left, right) => left === right,
      )
      const terminalizations = yield* Ref.make(0)
      const failedAdmission = yield* Effect.exit(coordinator.admit(Effect.succeed({
        key: "failed-commit",
        whenIdle: Effect.succeed(Option.some({
          commit: Effect.die("commit defect"),
          operation: Effect.void,
          terminalize: () => Ref.update(terminalizations, (count) => count + 1),
        })),
      })))
      expect(Exit.isFailure(failedAdmission)).toBe(true)
      expect(yield* Ref.get(terminalizations)).toBe(1)

      const next = yield* coordinator.admit(Effect.succeed({
        key: "next",
        whenIdle: Effect.succeed(Option.none()),
      }))
      expect(next._tag).toBe("Satisfied")
    })))
  })
})
