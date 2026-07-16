import { Data, Option, Schema } from "effect"

export const LlamaCliOperation = Schema.Literal("profile", "help", "list-devices", "fit")
export type LlamaCliOperation = Schema.Schema.Type<typeof LlamaCliOperation>

export const LlamaCliFailureReason = Schema.Literal("invalid-input", "unsupported", "command-failed", "invalid-output")
export type LlamaCliFailureReason = Schema.Schema.Type<typeof LlamaCliFailureReason>

export class LlamaCliError extends Data.TaggedError("LlamaCliError")<{
  readonly operation: LlamaCliOperation
  readonly reason: LlamaCliFailureReason
  readonly field: Option.Option<string>
}> {
  static make(operation: LlamaCliOperation, reason: LlamaCliFailureReason, field: Option.Option<string> = Option.none()): LlamaCliError {
    return new LlamaCliError({ operation, reason, field })
  }
}
