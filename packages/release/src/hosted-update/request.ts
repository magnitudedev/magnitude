import { Effect, Option, Schema } from "effect"
import { isValidVersion } from "../client-update/release-channels"
import { RequestNonce } from "./request-auth"
import { LinuxPackageFormat } from "../linux-package"

export const Version = Schema.String.pipe(Schema.maxLength(96), Schema.filter(isValidVersion))
const PlatformVersion = Schema.String.pipe(Schema.minLength(1), Schema.maxLength(96), Schema.pattern(/^[a-zA-Z0-9 ._+()-]+$/))
export const UpdateCheckReason = Schema.Literal("launch", "scheduled", "manual")
export type UpdateCheckReason = typeof UpdateCheckReason.Type
/** Why a recorded installation attempt failed, as classified where the failure happened. */
export const UpdateFailureReason = Schema.Literal("verify", "install", "authorization", "startup")
export type UpdateFailureReason = typeof UpdateFailureReason.Type
export const UpdateOutcomeReason = Schema.Union(UpdateFailureReason, Schema.Literal("incomplete"))
export const UpdateOutcomeKind = Schema.Literal("applied", "failed", "deferred")
/** Which application owner made the request: the desktop app, `serve` run by hand, or the system service. */
export const UpdateOwner = Schema.Literal("desktop", "headless", "service")
export type UpdateOwner = typeof UpdateOwner.Type
/**
 * An installation attempt's result. `version` is the release attempted and `from` the version that
 * was running when it was prepared; `from` is absent only for preparations recorded by older clients.
 */
export const UpdateOutcome = Schema.Struct({
  outcome: UpdateOutcomeKind,
  version: Version,
  from: Schema.optionalWith(Version, { as: "Option", exact: true }),
  reason: Schema.optionalWith(UpdateOutcomeReason, { as: "Option", exact: true }),
}).pipe(Schema.filter(outcome => (outcome.outcome === "failed") === Option.isSome(outcome.reason)))
export type UpdateOutcome = typeof UpdateOutcome.Type
const Label = Schema.String.pipe(Schema.minLength(1), Schema.maxLength(96), Schema.pattern(/^[\x20-\x7E]+$/))
const Reason = Schema.String.pipe(Schema.minLength(1), Schema.maxLength(200), Schema.pattern(/^[\x20-\x7E]+$/))
const Count = Schema.NumberFromString.pipe(Schema.int(), Schema.nonNegative())
export const UpdateHardwareFields = Schema.Struct({
  backend: Schema.optionalWith(Schema.Literal("cpu", "metal", "cuda", "vulkan"), { as: "Option", exact: true }),
  gpu_name: Schema.optionalWith(Label, { as: "Option", exact: true }),
  gpu_kind: Schema.optionalWith(Schema.Literal("discrete", "integrated"), { as: "Option", exact: true }),
  gpu_memory_gb: Schema.optionalWith(Count, { as: "Option", exact: true }),
  gpu_count: Schema.optionalWith(Count, { as: "Option", exact: true }),
  gpu_unused_reason: Schema.optionalWith(Reason, { as: "Option", exact: true }),
  cpu_model: Schema.optionalWith(Label, { as: "Option", exact: true }),
  cpu_cores: Schema.optionalWith(Count, { as: "Option", exact: true }),
  memory_gb: Schema.optionalWith(Count, { as: "Option", exact: true }),
})
export const UpdateHardware = UpdateHardwareFields.pipe(Schema.filter(hardware => Option.isSome(hardware.gpu_name) || (Option.isNone(hardware.gpu_kind) && Option.isNone(hardware.gpu_memory_gb))))
export type UpdateHardware = typeof UpdateHardware.Type
export const UpdateRequestFields = Schema.Struct({
  protocol: Schema.Literal("1"),
  product: Schema.Literal("desktop"),
  version: Version,
  os: Schema.Literal("darwin", "windows", "linux"),
  os_version: PlatformVersion,
  arch: Schema.Literal("arm64", "x64"),
  package: Schema.Union(Schema.Literal("mac-zip", "windows-exe"), LinuxPackageFormat),
  channel: Schema.Literal("stable", "beta", "alpha"),
  ts: Schema.NumberFromString.pipe(Schema.int(), Schema.nonNegative()),
  nonce: RequestNonce,
  distro: Schema.optionalWith(PlatformVersion, { as: "Option", exact: true }),
  distro_version: Schema.optionalWith(PlatformVersion, { as: "Option", exact: true }),
  owner: Schema.optionalWith(UpdateOwner, { as: "Option", exact: true }),
  reason: Schema.optionalWith(UpdateCheckReason, { as: "Option", exact: true }),
  outcome: Schema.optionalWith(UpdateOutcomeKind, { as: "Option", exact: true }),
  outcome_version: Schema.optionalWith(Version, { as: "Option", exact: true }),
  outcome_reason: Schema.optionalWith(UpdateOutcomeReason, { as: "Option", exact: true }),
  outcome_from: Schema.optionalWith(Version, { as: "Option", exact: true }),
  backend: UpdateHardwareFields.fields.backend,
  gpu_name: UpdateHardwareFields.fields.gpu_name,
  gpu_kind: UpdateHardwareFields.fields.gpu_kind,
  gpu_memory_gb: UpdateHardwareFields.fields.gpu_memory_gb,
  gpu_count: UpdateHardwareFields.fields.gpu_count,
  gpu_unused_reason: UpdateHardwareFields.fields.gpu_unused_reason,
  cpu_model: UpdateHardwareFields.fields.cpu_model,
  cpu_cores: UpdateHardwareFields.fields.cpu_cores,
  memory_gb: UpdateHardwareFields.fields.memory_gb,
})
export const UpdateRequest = UpdateRequestFields.pipe(Schema.filter(request => (Option.isNone(request.outcome)
  ? Option.isNone(request.outcome_version) && Option.isNone(request.outcome_reason) && Option.isNone(request.outcome_from)
  : Option.isSome(request.outcome_version) && (request.outcome.value === "failed") === Option.isSome(request.outcome_reason))
  && (Option.isSome(request.gpu_name) || (Option.isNone(request.gpu_kind) && Option.isNone(request.gpu_memory_gb)))))
export type UpdateRequest = typeof UpdateRequest.Type
export class InvalidUpdateRequest extends Schema.TaggedError<InvalidUpdateRequest>()("InvalidUpdateRequest", {}) {}
export class ExpiredUpdateRequest extends Schema.TaggedError<ExpiredUpdateRequest>()("ExpiredUpdateRequest", {}) {}

export const decodeUpdateRequest = (url: URL, nowSeconds: number) => Effect.gen(function* () {
  if (url.href.length > 4096) return yield* new InvalidUpdateRequest()
  const fields: Record<string, string> = Object.create(null)
  for (const [key, value] of url.searchParams) {
    if (Object.hasOwn(fields, key)) return yield* new InvalidUpdateRequest()
    fields[key] = value
  }
  const request = yield* Schema.decodeUnknown(UpdateRequest)(fields, { onExcessProperty: "error" }).pipe(Effect.mapError(() => new InvalidUpdateRequest()))
  if (Math.abs(nowSeconds - request.ts) > 300) return yield* new ExpiredUpdateRequest()
  const compatible = request.os === "darwin" ? request.package === "mac-zip"
    : request.os === "windows" ? request.package === "windows-exe" && request.arch === "x64"
    : Schema.is(LinuxPackageFormat)(request.package)
  if (!compatible) return yield* new InvalidUpdateRequest()
  return request
})
