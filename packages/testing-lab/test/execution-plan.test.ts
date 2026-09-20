import { Effect, Option, Schema } from "effect"
import { expect, test } from "vitest"
import { planRun, targets } from "../src/catalog"
import { orderCases } from "../src/case-runner"
import { RunRequest } from "../src/domain"
import { ExecutionPlan, planExecution } from "../src/execution-plan"

const request = (kind: "source" | "artifacts") => Schema.decodeUnknownSync(RunRequest)({ schemaVersion: 1, idempotencyKey: "execution-plan-fixture",
  owner: "fixture", input: { kind, digest: "a".repeat(64) }, selection: { kind: "profile", profile: "full" }, mode: "verify", trust: "developer",
  allowSpark: false, limits: { concurrency: 4, deadlineMinutes: 120, budgetUsd: 1000, idleMinutes: 15 } })

test("the complete source matrix shares native builds while preserving every consumer and case", () => Effect.runPromise(Effect.gen(function* () {
  const plan = yield* planRun(request("source"))
  const work = yield* planExecution(plan.request, plan.targets, targets)
  const encoded = yield* Schema.encode(Schema.parseJson(ExecutionPlan))(work)
  expect(yield* Schema.decodeUnknown(Schema.parseJson(ExecutionPlan))(encoded)).toEqual(work)
  const builds = work.filter(item => item.kind === "build"), tests = work.filter(item => item.kind === "test")
  expect(builds).toHaveLength(8)
  expect(tests).toHaveLength(44)
  expect(new Set(work.map(item => item.id)).size).toBe(52)
  expect(builds.every(item => item.target.target.backend === "cpu" && item.target.target.provider !== "spark")).toBe(true)
  for (const consumer of tests) {
    const producer = builds.find(item => item.id === Option.getOrThrow(consumer.producer))!
    expect(producer.consumers).toContain(consumer.target.target.id)
    expect(producer.target.target.artifactHost).toBe(consumer.target.target.artifactHost)
    expect(producer.backend).toBe(consumer.target.target.backend)
    const original = plan.targets.find(target => target.target.id === consumer.target.target.id)!
    expect([...producer.target.cases, ...consumer.target.cases].map(test => `${test.id}/${Option.getOrElse(test.harness, () => "")}`).sort())
      .toEqual(original.cases.map(test => `${test.id}/${Option.getOrElse(test.harness, () => "")}`).sort())
    yield* orderCases(consumer.target.cases)
  }
  expect(builds.find(item => item.id === "build:linux-arm64-gnu:cuda")!.target.blockers).not.toHaveLength(0)
  expect(tests.find(item => item.target.target.provider === "spark")!.target.blockers).not.toHaveLength(0)
  const reversed = yield* planExecution(plan.request, [...plan.targets].reverse(), targets)
  expect(reversed.filter(item => item.kind === "build")).toEqual(builds)
})))

test("artifact-only runs preserve packaging verification and create no producer", () => Effect.runPromise(Effect.gen(function* () {
  const plan = yield* planRun(request("artifacts"))
  const work = yield* planExecution(plan.request, plan.targets, targets)
  expect(work).toHaveLength(44)
  for (const item of work) {
    expect(item.kind).toBe("test")
    if (item.kind === "test") expect(Option.isNone(item.producer)).toBe(true)
    expect(item.target).toEqual(plan.targets.find(target => target.target.id === item.target.target.id))
  }
})))
