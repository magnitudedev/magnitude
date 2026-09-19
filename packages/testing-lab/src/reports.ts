import { FileSystem } from "@effect/platform"
import { Effect, Option, Schema } from "effect"
import { dirname, resolve } from "node:path"
import { InvalidInput, RunResult } from "./domain"

const xml = (value: string) => Array.from(value, char => {
  const code = char.codePointAt(0)!
  return code === 9 || code === 10 || code === 13 || code >= 32 && code <= 0xd7ff || code >= 0xe000 && code <= 0xfffd || code >= 0x10000 ? char : "�"
}).join("").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&apos;")

/** Selected cases must all be represented; missing coverage and cleanup cannot produce green JUnit. */
export const junitReport = (result: RunResult): string => {
  const cases: string[] = []
  let failures = 0, errors = 0
  const expected = new Set<string>()
  const key = (target: string, id: string, harness: Option.Option<string>) => `${target}/${id}/${Option.getOrElse(harness, () => "")}`
  const error = (name: string, detail: string) => {
    errors++
    cases.push(`<testcase classname="lab.infrastructure" name="${xml(name)}"><error message="${xml(detail)}">${xml(detail)}</error></testcase>`)
  }
  for (const target of result.plan.targets) for (const planned of target.cases) {
    const identity = key(target.target.id, planned.id, planned.harness)
    if (expected.has(identity)) { error(identity, "Plan contains duplicate selected case"); continue }
    expected.add(identity)
    const matches = result.cases.filter(test => key(test.targetId, test.caseId, test.harness) === identity)
    if (matches.length !== 1) { error(identity, matches.length === 0 ? "Selected case has no result" : "Selected case has duplicate results"); continue }
    const test = matches[0]!
    const status = test.outcome.status
    const detail = xml(test.outcome.detail)
    const duration = (Date.parse(test.endedAt) - Date.parse(test.startedAt)) / 1000
    const time = Number.isFinite(duration) && duration >= 0 ? ` time="${duration}"` : ""
    const outcome = status === "passed" ? "" : status === "failed"
      ? (failures++, `<failure type="${test.outcome.category}" message="${detail}">${detail}</failure>`)
      : (errors++, `<error type="${status}" message="${detail}">${detail}</error>`)
    const evidence = xml(JSON.stringify({ detail: test.outcome.detail, evidence: test.evidence }))
    cases.push(`<testcase classname="${xml(`${target.target.id}.${planned.suite}.${Option.getOrElse(planned.harness, () => "shared")}`)}" name="${xml(`${planned.id}: ${planned.title}`)}"${time}>${outcome}<system-out>${evidence}</system-out></testcase>`)
  }
  for (const test of result.cases) if (!expected.has(key(test.targetId, test.caseId, test.harness))) error(key(test.targetId, test.caseId, test.harness), "Result was not selected by the admitted plan")
  for (const [index, detail] of result.cleanupErrors.entries()) error(`cleanup-${index + 1}`, detail)
  if (expected.size === 0) error("coverage", "Run selected no test cases")
  return `<?xml version="1.0" encoding="UTF-8"?>\n<testsuites tests="${cases.length}" failures="${failures}" errors="${errors}"><testsuite name="${xml(result.runId)}" tests="${cases.length}" failures="${failures}" errors="${errors}">\n${cases.join("\n")}\n</testsuite></testsuites>\n`
}

export const ReportPaths = Schema.Struct({
  json: Schema.optionalWith(Schema.NonEmptyString, { as: "Option", exact: true }),
  junit: Schema.optionalWith(Schema.NonEmptyString, { as: "Option", exact: true }),
})
export const writeReports = (result: RunResult, paths: typeof ReportPaths.Type) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  if (Option.isSome(paths.json) && Option.isSome(paths.junit) && resolve(paths.json.value) === resolve(paths.junit.value)) return yield* new InvalidInput({ message: "JSON and JUnit output paths must differ" })
  const write = (path: string, contents: string) => Effect.gen(function* () {
    const destination = resolve(path)
    yield* fs.makeDirectory(dirname(destination), { recursive: true })
    const temporary = `${destination}.${crypto.randomUUID()}.tmp`
    yield* Effect.acquireUseRelease(fs.writeFileString(temporary, contents, { mode: 0o600, flag: "wx" }),
      () => fs.rename(temporary, destination), () => fs.remove(temporary, { force: true }).pipe(Effect.orDie))
  })
  if (Option.isSome(paths.json)) yield* write(paths.json.value, yield* Schema.encode(Schema.parseJson(RunResult))(result))
  if (Option.isSome(paths.junit)) yield* write(paths.junit.value, junitReport(result))
})
