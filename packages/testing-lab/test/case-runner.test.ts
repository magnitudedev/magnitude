import { expect, test } from "vitest"
import { Effect, Option } from "effect"
import { CaseExecutor, orderCases, runCases } from "../src/case-runner"
import { AssertionFailure, CaseId, InfrastructureFailure, type PlannedCase } from "../src/domain"
import { targets } from "../src/catalog"
const c = (id: string, dependencies: string[] = [], harness: "pi" | "hermes" | undefined = undefined): PlannedCase => ({ id: CaseId.make(id), suite: id.startsWith("H") ? "harness" : "app", title: id,
  prerequisites: dependencies.map(d => CaseId.make(d)), harness: Option.fromNullable(harness), timeoutSeconds: 5 })
test("orders prerequisites and keeps harness occurrences separate", async () => {
  const ordered = await Effect.runPromise(orderCases([c("H2", ["H1"], "hermes"), c("H1", [], "pi"), c("H1", [], "hermes")]))
  expect(ordered.map(t => `${t.id}:${Option.getOrThrow(t.harness)}`)).toEqual(["H1:hermes", "H2:hermes", "H1:pi"])
})
test.each([[c("A1", ["A2"]), c("A2", ["A1"])], [c("A1", ["A2"])], [c("A1"), c("A1")]])("rejects invalid dependency graphs before execution", async (...cases) => {
  expect((await Effect.runPromise(orderCases(cases).pipe(Effect.either)))._tag).toBe("Left")
})
test("assertion failure blocks dependants but still executes independent cases", async () => {
  const executed: string[] = []
  const results = await Effect.runPromise(runCases(targets[0]!, [c("A2", ["A1"]), c("A1"), c("A3")]).pipe(Effect.provideService(CaseExecutor, {
    execute: t => { executed.push(t.id); return t.id === "A1" ? Effect.fail(new AssertionFailure({ message: "Actual assertion failed" })) : Effect.succeed({ detail: "Observed success", evidence: [] }) },
  })))
  expect(executed).toEqual(["A1", "A3"])
  expect(results.map(r => r.outcome.status)).toEqual(["blocked", "failed", "passed"])
})
test("infrastructure failure is blocked rather than passed or a product failure", async () => {
  const results = await Effect.runPromise(runCases(targets[0]!, [c("A1")]).pipe(Effect.provideService(CaseExecutor, {
    execute: () => Effect.fail(new InfrastructureFailure({ operation: "provider", message: "Quota unavailable" })),
  })))
  expect(results[0]?.outcome.status).toBe("blocked")
})
