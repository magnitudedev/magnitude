import { Data } from "effect"

// ── Binary errors ──

/** No `llama-server` binary found after exhausting all resolution tiers. */
export class LlamaCppBinaryNotFound extends Data.TaggedError("LlamaCppBinaryNotFound")<{
  readonly searched: readonly string[]
}> {}

/** Binary found but its build number is below the minimum supported version. */
export class LlamaCppBinaryVersionTooOld extends Data.TaggedError("LlamaCppBinaryVersionTooOld")<{
  readonly path: string
  readonly actual: number
  readonly minimum: number
}> {}

/** Failed to download the binary from GitHub releases. */
export class LlamaCppBinaryDownloadFailed extends Data.TaggedError("LlamaCppBinaryDownloadFailed")<{
  readonly url: string
  readonly reason: string
}> {}

/** Binary found but `--version` failed or output could not be parsed. */
export class LlamaCppBinaryValidationFailed extends Data.TaggedError("LlamaCppBinaryValidationFailed")<{
  readonly path: string
  readonly reason: string
}> {}

/** Current platform/arch has no compatible release asset. */
export class LlamaCppUnsupportedPlatform extends Data.TaggedError("LlamaCppUnsupportedPlatform")<{
  readonly platform: string
  readonly arch: string
}> {}

// ── Hardware errors ──

/** Generic hardware detection or fit assessment failure. */
export class LlamaCppHardwareError extends Data.TaggedError("LlamaCppHardwareError")<{
  readonly reason: string
}> {}

// ── Model errors ──

/** Requested model ID was not found on disk or in any running instance. */
export class LlamaCppModelNotFound extends Data.TaggedError("LlamaCppModelNotFound")<{
  readonly modelId: string
}> {}

/** Model download from HuggingFace failed. */
export class LlamaCppModelDownloadFailed extends Data.TaggedError("LlamaCppModelDownloadFailed")<{
  readonly repo: string
  readonly file: string
  readonly reason: string
}> {}

/** Access denied to a gated HuggingFace model repository. */
export class LlamaCppGatedModelAccessDenied extends Data.TaggedError("LlamaCppGatedModelAccessDenied")<{
  readonly repo: string
  readonly message: string
}> {}

/** HuggingFace token required but not configured. */
export class LlamaCppHfTokenMissing extends Data.TaggedError("LlamaCppHfTokenMissing")<{
  readonly repo: string
}> {}

// ── Server / instance errors ──

/** `llama-server` process failed to start. */
export class LlamaCppServerStartFailed extends Data.TaggedError("LlamaCppServerStartFailed")<{
  readonly reason: string
  readonly stderr?: string
}> {}

/** Server process started but did not become healthy within the timeout. */
export class LlamaCppServerTimeout extends Data.TaggedError("LlamaCppServerTimeout")<{
  readonly endpoint: string
  readonly phase: "bind" | "load" | "health"
}> {}

/** Server process ran out of memory while loading a model. */
export class LlamaCppServerOutOfMemory extends Data.TaggedError("LlamaCppServerOutOfMemory")<{
  readonly attempted: { readonly ngl: number; readonly ctx: number }
  readonly stderr: string
}> {}

/** HTTP error communicating with a running `llama-server` endpoint. */
export class LlamaCppEndpointError extends Data.TaggedError("LlamaCppEndpointError")<{
  readonly reason: string
}> {}

/** No free port available for a managed server. */
export class LlamaCppPortUnavailable extends Data.TaggedError("LlamaCppPortUnavailable")<{
  readonly port: number
}> {}

// ── Detection errors ──

/** Failed to detect or fingerprint a running server. */
export class LlamaCppDetectionFailed extends Data.TaggedError("LlamaCppDetectionFailed")<{
  readonly reason: string
}> {}
