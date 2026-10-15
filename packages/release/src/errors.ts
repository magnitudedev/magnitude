import { Data, type Option } from "effect"

export class ReleaseAcquisitionError extends Data.TaggedError("ReleaseAcquisitionError")<{
  readonly stage: "download" | "validate" | "archive" | "install" | "verify"
  readonly message: string
  readonly transient: boolean
}> {}

/** The disk holding a release installation lacks the space for it; retrying can't help until space is freed. */
export class ReleaseDiskFullError extends Data.TaggedError("ReleaseDiskFullError")<{
  readonly directory: string
  readonly requiredBytes: Option.Option<number>
  readonly availableBytes: Option.Option<number>
}> {}
