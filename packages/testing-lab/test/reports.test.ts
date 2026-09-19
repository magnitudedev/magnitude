import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Option, Schema } from "effect"
import { join } from "node:path"
import { expect, test } from "vitest"
import { planRun } from "../src/catalog"
import { RunRequest, RunResult } from "../src/domain"
import { junitReport, ReportPaths, writeReports } from "../src/reports"

const fixture = Effect.gen(function* () {
  const request = yield* Schema.decodeUnknown(RunRequest)({ schemaVersion: 1, idempotencyKey: "report-fixture", owner: "developer", input: { kind: "source", digest: "a".repeat(64) },
    selection: { kind: "custom", targets: ["ubuntu-24.04-x64-cpu-intel"], suites: ["cli"], harnesses: ["pi"] }, mode: "verify", trust: "developer", allowSpark: false,
    limits: { concurrency: 1, deadlineMinutes: 60, budgetUsd: 50, idleMinutes: 15 } })
  const plan = yield* planRun(request)
  return RunResult.make({ schemaVersion: 1, runId: RunResult.fields.runId.make("run-00000000-0000-0000-0000-000000000001"), plan,
    startedAt: "2026-09-18T10:00:00Z", endedAt: "2026-09-18T10:01:00Z", cleanupErrors: [],
    cases: plan.targets.flatMap(target => target.cases.map(test => ({ targetId: target.target.id, caseId: test.id, harness: test.harness,
      startedAt: "2026-09-18T10:00:00Z", endedAt: "2026-09-18T10:00:02Z", outcome: { status: "passed" as const, detail: "Accepted" }, evidence: [] }))),
  })
})
test("passed reports account for every selected case", async () => {
  const result = await Effect.runPromise(fixture)
  const xml = junitReport(result)
  expect(xml).toContain(`tests="${result.cases.length}" failures="0" errors="0"`)
  expect(xml.match(/<testcase /g)).toHaveLength(result.cases.length)
  expect(xml).toContain('time="2"')
})
test.each(["blocked", "cancelled", "not-selected"] as const)("selected %s cases are JUnit errors", async status => {
  const result = await Effect.runPromise(fixture)
  const xml = junitReport({ ...result, cases: [{ ...result.cases[0]!, outcome: { status, detail: "Unqualified" } }, ...result.cases.slice(1)] })
  expect(xml).toContain('failures="0" errors="1"')
  expect(xml).toContain(`<error type="${status}"`)
})
test("product failures remain failures with XML-safe diagnostics", async () => {
  const result = await Effect.runPromise(fixture)
  const xml = junitReport({ ...result, cases: [{ ...result.cases[0]!, outcome: { status: "failed", category: "app", detail: '<bad a="b">&\u0000\ud800' } }, ...result.cases.slice(1)] })
  expect(xml).toContain('failures="1" errors="0"')
  expect(xml).toContain('&lt;bad a=&quot;b&quot;&gt;&amp;��')
  expect(xml).not.toContain('\u0000')
})
test("missing, duplicate and cleanup results cannot look green", async () => {
  const result = await Effect.runPromise(fixture)
  expect(junitReport({ ...result, cases: result.cases.slice(1) })).toContain('errors="1"')
  expect(junitReport({ ...result, cases: [...result.cases, result.cases[0]!] })).toContain('errors="1"')
  expect(junitReport({ ...result, cleanupErrors: ["VM still allocated"] })).toContain('errors="1"')
  expect(junitReport({ ...result, plan: { ...result.plan, targets: [] }, cases: [] })).toContain('errors="1"')
})
test("writes exact JSON and JUnit and rejects colliding report destinations", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-reports-" })
  const result = yield* fixture
  const paths = ReportPaths.make({ json: Option.some(join(root, "out", "run.json")), junit: Option.some(join(root, "out", "junit.xml")) })
  yield* writeReports(result, paths)
  const read = yield* Schema.decodeUnknown(Schema.parseJson(RunResult))(yield* fs.readFileString(Option.getOrThrow(paths.json)))
  expect(read).toEqual(result)
  expect(yield* fs.readFileString(Option.getOrThrow(paths.junit))).toBe(junitReport(result))
  const collision = yield* writeReports(result, { json: paths.json, junit: paths.json }).pipe(Effect.either)
  expect(collision._tag).toBe("Left")
  expect((yield* fs.readDirectory(join(root, "out"))).sort()).toEqual(["junit.xml", "run.json"])
})).pipe(Effect.provide(BunContext.layer))))
