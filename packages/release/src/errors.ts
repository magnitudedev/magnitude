import { Data } from "effect"

export class ReleaseAcquisitionError extends Data.TaggedError("ReleaseAcquisitionError")<{
  readonly stage: "download" | "authenticate" | "archive" | "install" | "verify"
  readonly message: string
  readonly transient: boolean
}> {}
