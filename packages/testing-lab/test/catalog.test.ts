import { describe, expect, test } from "vitest"
import { Effect, Option, Schema } from "effect"
import { cases, planRun, prTargetIds, targets } from "../src/catalog"
import { RunRequest } from "../src/domain"

export const request = (selection: unknown = { kind: "profile", profile: "pr" }) => Schema.decodeUnknownSync(RunRequest)({
  schemaVersion: 1, idempotencyKey: "test-request-001", owner: "developer", input: { kind: "source", digest: "a".repeat(64) },
  selection, mode: "verify", trust: "developer", allowSpark: false,
  limits: { concurrency: 4, deadlineMinutes: 120, budgetUsd: 1000, idleMinutes: 15 },
})

describe("coverage policy", () => {
  test("expands every agreed combination without excluded platforms", () => {
    expect(targets).toHaveLength(44)
    expect(new Set(targets.map(t => t.id)).size).toBe(44)
    expect(new Set(cases.map(c => c.id)).size).toBe(52)
    expect(new Set(cases.map(c => c.suite)).size).toBe(9)
    expect(targets.filter(t => t.backend === "cuda")).toHaveLength(12)
    expect(targets.every(t => t.os !== "macos" || t.arch === "arm64")).toBe(true)
  })
  test("PR keeps all 12 representatives and blocks office access instead of omitting it", async () => {
    const plan = await Effect.runPromise(planRun(request()))
    expect(plan.targets.map(p => p.target.id)).toEqual(prTargetIds)
    expect(plan.targets.find(p => p.target.provider === "spark")?.blockers).not.toHaveLength(0)
    expect(plan.artifactHosts).toHaveLength(4)
    for (const p of plan.targets) expect(p.cases.filter(c => c.id === "H5").map(c => Option.getOrThrow(c.harness))).toEqual(["pi", "opencode", "hermes"])
  })
  test("full retains all desired targets even before qualification", async () => {
    const plan = await Effect.runPromise(planRun(request({ kind: "profile", profile: "full" })))
    expect(plan.targets).toHaveLength(44)
    expect(plan.targets.every(t => t.cases.filter(c => c.id === "H7").length === 3)).toBe(true)
  })
  test("custom harness selection expands real setup dependencies and device attestation", async () => {
    const plan = await Effect.runPromise(planRun(request({ kind: "custom", targets: ["ubuntu-24.04-x64-cuda-a10"], suites: ["harness"], harnesses: ["hermes"] })))
    expect(plan.targets[0]!.cases.map(c => c.id)).toEqual(expect.arrayContaining(["P1", "I2", "A3", "A5", "E6", "H5"]))
    expect(plan.targets[0]!.cases.filter(c => c.suite === "harness").every(c => Option.getOrThrow(c.harness) === "hermes")).toBe(true)
  })
  test("does not allow a full or release claim to be narrowed implicitly", async () => {
    const result = await Effect.runPromise(planRun(request({ kind: "profile", profile: "full", target: targets[0]!.id })).pipe(Effect.either))
    expect(result._tag).toBe("Left")
  })
  test("release refuses source, warm reuse, or untrusted execution", async () => {
    expect(await Effect.runPromise(planRun(request({ kind: "profile", profile: "release" })).pipe(Effect.either))).toMatchObject({ _tag: "Left" })
  })
  test("prerequisites form an acyclic graph", () => {
    const visit = (id: string, ancestors: Set<string>): void => {
      expect(ancestors.has(id)).toBe(false)
      const found = cases.find(c => c.id === id)
      expect(found).toBeDefined()
      for (const prerequisite of found!.prerequisites) visit(prerequisite, new Set([...ancestors, id]))
    }
    for (const c of cases) visit(c.id, new Set())
  })
})
