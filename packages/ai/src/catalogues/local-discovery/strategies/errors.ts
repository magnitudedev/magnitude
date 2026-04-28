import { Data } from "effect"

export class LocalDiscoveryError extends Data.TaggedError("LocalDiscoveryError")<{
  readonly message: string
  readonly cause?: unknown
}> {}