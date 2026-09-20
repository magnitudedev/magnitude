import { Option, Schema } from "effect"

export const RunId = Schema.String.pipe(Schema.pattern(/^run-[a-f0-9-]{36}$/), Schema.brand("LabRunId"))
export type RunId = typeof RunId.Type
export const LeaseId = Schema.String.pipe(Schema.pattern(/^lease-[a-f0-9-]{36}$/), Schema.brand("LabLeaseId"))
export type LeaseId = typeof LeaseId.Type
export const TargetId = Schema.NonEmptyString.pipe(Schema.pattern(/^[a-z0-9][a-z0-9.-]+$/), Schema.brand("LabTargetId"))
export type TargetId = typeof TargetId.Type
export const CaseId = Schema.String.pipe(Schema.pattern(/^[PIAEHRCUX][1-7]$/), Schema.brand("LabCaseId"))
export type CaseId = typeof CaseId.Type
export const Digest = Schema.String.pipe(Schema.pattern(/^[a-f0-9]{64}$/), Schema.brand("Sha256"))
export type Digest = typeof Digest.Type
export const ObjectBatch = Schema.Array(Digest).pipe(Schema.maxItems(1000))
export const OwnerId = Schema.NonEmptyString.pipe(Schema.brand("LabOwnerId"))
export const IdempotencyKey = Schema.String.pipe(Schema.minLength(8), Schema.maxLength(200), Schema.brand("LabIdempotencyKey"))
export const Backend = Schema.Literal("cpu", "metal", "cuda")
export const Architecture = Schema.Literal("arm64", "x64")
export const Hardware = Schema.Literal("apple-silicon", "intel", "amd", "arm", "a10", "rtx-pro-6000", "dgx-spark")
export const Provider = Schema.Literal("azure", "namespace", "spark", "local")
export const Suite = Schema.Literal("package", "install", "app", "endpoint", "harness", "recovery", "cli", "update", "uninstall")
export type Suite = typeof Suite.Type
export const Harness = Schema.Literal("pi", "opencode", "hermes")
export type Harness = typeof Harness.Type
export const Mode = Schema.Literal("iterate", "verify")
export const Profile = Schema.Literal("quick", "pr", "full", "release")
export const Trust = Schema.Literal("developer", "trusted-ci", "untrusted-ci")
export const Principal = Schema.Struct({ owner: OwnerId, trust: Trust })
export type Principal = typeof Principal.Type
export const Target = Schema.Struct({
  id: TargetId, os: Schema.Literal("macos", "windows", "ubuntu", "debian", "fedora", "redhat", "dgx-os"),
  version: Schema.NonEmptyString, arch: Architecture, backend: Backend, hardware: Hardware,
  provider: Provider, artifactHost: Schema.Literal("darwin-arm64", "windows-x64-msvc", "linux-x64-gnu", "linux-arm64-gnu"),
  packageFormat: Schema.Literal("dmg", "exe", "deb", "rpm"),
})
export type Target = typeof Target.Type

export const Input = Schema.Union(
  Schema.Struct({ kind: Schema.Literal("source"), digest: Digest }),
  Schema.Struct({ kind: Schema.Literal("artifacts"), digest: Digest }),
)
export const Selection = Schema.Union(
  Schema.Struct({ kind: Schema.Literal("profile"), profile: Profile,
    target: Schema.optionalWith(TargetId, { as: "Option", exact: true }),
    harnesses: Schema.optionalWith(Schema.NonEmptyArray(Harness), { as: "Option", exact: true }) }),
  Schema.Struct({ kind: Schema.Literal("custom"), targets: Schema.NonEmptyArray(TargetId),
    suites: Schema.NonEmptyArray(Suite), harnesses: Schema.NonEmptyArray(Harness) }),
)
export const Limits = Schema.Struct({
  concurrency: Schema.Int.pipe(Schema.between(1, 32)),
  deadlineMinutes: Schema.Int.pipe(Schema.between(1, 1440)),
  budgetUsd: Schema.Number.pipe(Schema.positive(), Schema.finite()),
  idleMinutes: Schema.Int.pipe(Schema.between(1, 60)),
})
export const RunRequest = Schema.Struct({
  schemaVersion: Schema.Literal(1), idempotencyKey: IdempotencyKey, owner: OwnerId,
  input: Input, updateFrom: Schema.optionalWith(Schema.Struct({ kind: Schema.Literal("artifacts"), digest: Digest }), { as: "Option", exact: true }), selection: Selection, mode: Mode, trust: Trust, limits: Limits,
  allowSpark: Schema.Boolean,
})
export type RunRequest = typeof RunRequest.Type
export const runInputs = (request: RunRequest): readonly (typeof Input.Type)[] => [request.input, ...Option.toArray(request.updateFrom)]

export const PlannedCase = Schema.Struct({
  id: CaseId, suite: Suite, title: Schema.String, timeoutSeconds: Schema.Int.pipe(Schema.positive()),
  harness: Schema.optionalWith(Harness, { as: "Option", exact: true }),
  prerequisites: Schema.Array(CaseId),
})
export type PlannedCase = typeof PlannedCase.Type
export const TargetPlan = Schema.Struct({ target: Target, cases: Schema.Array(PlannedCase), blockers: Schema.Array(Schema.String) })
export const RunPlan = Schema.Struct({ schemaVersion: Schema.Literal(1), request: RunRequest,
  targets: Schema.Array(TargetPlan), artifactHosts: Schema.Array(Schema.String), estimatedComputeUsd: Schema.Number,
})
export type RunPlan = typeof RunPlan.Type

export const Evidence = Schema.Struct({ path: Schema.NonEmptyString, sha256: Digest, bytes: Schema.Int.pipe(Schema.nonNegative()) })
export const CaseOutcome = Schema.Union(
  Schema.Struct({ status: Schema.Literal("passed"), detail: Schema.String }),
  Schema.Struct({ status: Schema.Literal("failed"), category: Schema.Literal("build", "package", "app", "endpoint", "harness", "model-output"), detail: Schema.String }),
  Schema.Struct({ status: Schema.Literal("blocked"), detail: Schema.String }),
  Schema.Struct({ status: Schema.Literal("cancelled"), detail: Schema.String }),
  Schema.Struct({ status: Schema.Literal("not-selected"), detail: Schema.String }),
)
export const CaseResult = Schema.Struct({
  targetId: TargetId, caseId: CaseId, harness: Schema.optionalWith(Harness, { as: "Option", exact: true }),
  startedAt: Schema.String, endedAt: Schema.String, outcome: CaseOutcome, evidence: Schema.Array(Evidence),
})
export type CaseResult = typeof CaseResult.Type
export const RunResult = Schema.Struct({
  schemaVersion: Schema.Literal(1), runId: RunId, plan: RunPlan, cases: Schema.Array(CaseResult),
  cleanupErrors: Schema.Array(Schema.String), startedAt: Schema.String, endedAt: Schema.String,
})
export type RunResult = typeof RunResult.Type

export class InvalidInput extends Schema.TaggedError<InvalidInput>()("InvalidInput", { message: Schema.String }) {}
export class InfrastructureFailure extends Schema.TaggedError<InfrastructureFailure>()("InfrastructureFailure", {
  operation: Schema.String, message: Schema.String,
  evidence: Schema.optionalWith(Schema.Array(Evidence), { as: "Option", exact: true }).pipe(Schema.withConstructorDefault(() => Option.none())),
}) {}
export class AssertionFailure extends Schema.TaggedError<AssertionFailure>()("AssertionFailure", {
  message: Schema.String,
}) {}

/** Root product failure wins over dependent blocks. Cleanup is never hidden by green assertions. */
export const resultExitCode = (result: RunResult): 0 | 1 | 2 | 3 => {
  if (result.cases.some(c => c.outcome.status === "failed")) return 1
  if (result.cleanupErrors.length > 0 || result.cases.some(c => c.outcome.status === "blocked")) return 2
  if (result.cases.some(c => c.outcome.status === "cancelled")) return 3
  const expected = result.plan.targets.reduce((sum, target) => sum + target.cases.length, 0)
  if (expected === 0 || result.cases.length !== expected || result.cases.some(c => c.outcome.status !== "passed")) return 2
  const keys = result.cases.map(c => `${c.targetId}/${c.caseId}/${c.harness._tag === "Some" ? c.harness.value : ""}`)
  if (new Set(keys).size !== keys.length) return 2
  const selected = new Set(result.plan.targets.flatMap(t => t.cases.map(c => `${t.target.id}/${c.id}/${c.harness._tag === "Some" ? c.harness.value : ""}`)))
  if (keys.some(key => !selected.has(key))) return 2
  return 0
}
