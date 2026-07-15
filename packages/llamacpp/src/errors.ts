import { Data } from "effect"

export class DistributionInspectionError extends Data.TaggedError("DistributionInspectionError")<{
  readonly operation: "inspect"
  readonly reason: string
  readonly cause?: unknown
}> {}

export class DistributionInstallError extends Data.TaggedError("DistributionInstallError")<{
  readonly operation: "install"
  readonly code: "unsupported_platform" | "download_failed" | "integrity_failed" | "storage_failed"
  readonly stage: "resolving" | "downloading" | "extracting" | "verifying" | "publishing"
  readonly reason: string
  readonly cause?: unknown
}> {}

export class LlamaCppExecutableValidationError extends Data.TaggedError("LlamaCppExecutableValidationError")<{
  readonly path: string
  readonly reason: string
  readonly cause?: unknown
}> {}

export class LlamaCppHostError extends Data.TaggedError("LlamaCppHostError")<{
  readonly operation: "inspect" | "plan"
  readonly reason: string
  readonly cause?: unknown
}> {}

export class LlamaCppModelStoreError extends Data.TaggedError("LlamaCppModelStoreError")<{
  readonly operation: "inspect" | "resolve" | "download" | "delete"
  readonly code:
    | "artifact_not_found"
    | "artifact_not_owned"
    | "invalid_plan"
    | "insufficient_space"
    | "download_failed"
    | "integrity_failed"
    | "storage_failed"
  readonly reason: string
  readonly modelId?: string
  readonly cause?: unknown
}> {}

export class LlamaCppRuntimeError extends Data.TaggedError("LlamaCppRuntimeError")<{
  readonly operation: "inspect" | "ensure_serving"
  readonly code:
    | "distribution_unavailable"
    | "model_unavailable"
    | "external_unavailable"
    | "server_start_failed"
    | "server_timeout"
    | "identity_mismatch"
    | "context_mismatch"
    | "endpoint_failed"
  readonly reason: string
  readonly cause?: unknown
}> {}

export class LlamaCppEndpointClientError extends Data.TaggedError("LlamaCppEndpointClientError")<{
  readonly operation: "health" | "props" | "models"
  readonly endpoint: string
  readonly reason: string
  readonly cause?: unknown
}> {}
