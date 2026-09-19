import { Cause, Context, DateTime, Effect, Exit, Option, Schema } from "effect"
import { AssertionFailure, CaseId, CaseResult, Evidence, Harness, InfrastructureFailure, InvalidInput, PlannedCase, Target } from "./domain"

export const CaseObservation = Schema.Struct({ detail: Schema.String, evidence: Schema.Array(Evidence) })
export type CaseObservation = typeof CaseObservation.Type
export interface CaseExecutor {
  readonly execute: (test: PlannedCase) => Effect.Effect<CaseObservation, AssertionFailure | InfrastructureFailure>
}
export const CaseExecutor = Context.GenericTag<CaseExecutor>("@magnitudedev/testing-lab/CaseExecutor")
const key = (id: CaseId, harness: Option.Option<Harness>) => `${id}:${Option.getOrElse(harness, () => "")}`
const category = (test: PlannedCase): "build" | "package" | "app" | "endpoint" | "harness" =>
  test.id === "P1" ? "build" : test.suite === "package" || test.suite === "install" || test.suite === "uninstall" || test.suite === "update" ? "package"
    : test.suite === "endpoint" ? "endpoint" : test.suite === "harness" ? "harness" : "app"
const now = DateTime.now.pipe(Effect.map(DateTime.formatIso))

/** Test dependencies are evaluated per harness, with shared application prerequisites evaluated once. */
export const orderCases = (cases: readonly PlannedCase[]) => Effect.gen(function* () {
  const byKey = new Map(cases.map(c => [key(c.id, c.harness), c]))
  if (byKey.size !== cases.length) return yield* new InvalidInput({ message: "Duplicate test case identity" })
  const visiting = new Set<string>(), visited = new Set<string>(), ordered: PlannedCase[] = []
  const visit = (test: PlannedCase): Effect.Effect<void, InvalidInput> => Effect.gen(function* () {
    const identity = key(test.id, test.harness)
    if (visited.has(identity)) return
    if (visiting.has(identity)) return yield* new InvalidInput({ message: `Cyclic test dependency at ${identity}` })
    visiting.add(identity)
    for (const id of test.prerequisites) {
      const dependency = byKey.get(key(id, id.startsWith("H") ? test.harness : Option.none()))
      if (!dependency) return yield* new InvalidInput({ message: `Missing prerequisite ${id} for ${identity}` })
      yield* visit(dependency)
    }
    visiting.delete(identity); visited.add(identity); ordered.push(test)
  })
  for (const test of cases) yield* visit(test)
  return ordered
})

/** Return exactly one outcome per selected case. Failures block dependants, never independent checks. */
export const runCases = (target: Target, cases: readonly PlannedCase[]) => Effect.gen(function* () {
  const executor = yield* CaseExecutor
  const ordered = yield* orderCases(cases)
  return yield* Effect.uninterruptibleMask(restore => Effect.gen(function* () {
    const results = new Map<string, CaseResult>()
    let cancelled = false
    for (const test of ordered) {
      const startedAt = yield* now
      const dependencies = test.prerequisites.map(id => results.get(key(id, id.startsWith("H") ? test.harness : Option.none()))!)
      let outcome: CaseResult["outcome"], evidence: readonly (typeof Evidence.Type)[] = []
      if (cancelled) outcome = { status: "cancelled", detail: "Run was interrupted before this case" }
      else if (dependencies.some(d => d.outcome.status !== "passed")) outcome = { status: "blocked", detail: `Prerequisite did not pass: ${dependencies.filter(d => d.outcome.status !== "passed").map(d => d.caseId).join(", ")}` }
      else {
        const exit = yield* restore(Effect.suspend(() => executor.execute(test)).pipe(
          Effect.timeoutFail({ duration: test.timeoutSeconds * 1000, onTimeout: () => new AssertionFailure({ message: `Case ${test.id} exceeded ${test.timeoutSeconds} seconds` }) }),
        )).pipe(Effect.exit)
        if (Exit.isSuccess(exit)) { outcome = { status: "passed", detail: exit.value.detail }; evidence = exit.value.evidence }
        else if (Cause.isInterruptedOnly(exit.cause)) { cancelled = true; outcome = { status: "cancelled", detail: "Case was interrupted" } }
        else {
          const error = Cause.failureOption(exit.cause)
          if (Option.isSome(error) && error.value._tag === "InfrastructureFailure") outcome = { status: "blocked", detail: error.value.message }
          else outcome = { status: "failed", category: category(test), detail: Option.isSome(error) ? error.value.message : "Test executor defect; inspect worker diagnostics" }
        }
      }
      const result = CaseResult.make({ targetId: target.id, caseId: test.id, harness: test.harness, startedAt, endedAt: yield* now, outcome, evidence })
      results.set(key(test.id, test.harness), result)
    }
    return cases.map(c => results.get(key(c.id, c.harness))!)
  }))
})
