import { Data, Schema } from "effect"
import { ProcessGroupSchema } from "./schemas"

export class AcnProcessStoreUnavailable extends Data.TaggedError("AcnProcessStoreUnavailable")<{
  readonly path: string
  readonly message: string
}> {}

export class AcnProcessStoreInvalid extends Data.TaggedError("AcnProcessStoreInvalid")<{
  readonly path: string
  readonly message: string
}> {}

export class AcnProcessStoreBusy extends Data.TaggedError("AcnProcessStoreBusy")<{
  readonly path: string
}> {}

export type AcnOwnerStoreError =
  | AcnProcessStoreUnavailable
  | AcnProcessStoreInvalid

export type AcnProcessStoreError =
  | AcnOwnerStoreError
  | AcnProcessStoreBusy

const ProcessIdSchema = Schema.Number.pipe(Schema.int(), Schema.positive())

export class ExactProcessIdentityObservationFailed extends Schema.TaggedError<ExactProcessIdentityObservationFailed>()(
  "ExactProcessIdentityObservationFailed",
  { pid: ProcessIdSchema, message: Schema.String },
) {}

export class ProcessGroupObservationFailed extends Schema.TaggedError<ProcessGroupObservationFailed>()(
  "ProcessGroupObservationFailed",
  { group: ProcessGroupSchema, message: Schema.String },
) {}

export class ProcessGroupSignalPermissionDenied extends Schema.TaggedError<ProcessGroupSignalPermissionDenied>()(
  "ProcessGroupSignalPermissionDenied",
  { group: ProcessGroupSchema, message: Schema.String },
) {}

export class ProcessGroupSignalFailed extends Schema.TaggedError<ProcessGroupSignalFailed>()(
  "ProcessGroupSignalFailed",
  { group: ProcessGroupSchema, message: Schema.String },
) {}

export class ProcessGroupAbsenceUnproven extends Schema.TaggedError<ProcessGroupAbsenceUnproven>()(
  "ProcessGroupAbsenceUnproven",
  { group: ProcessGroupSchema },
) {}

export type ExactProcessObservationError =
  | ExactProcessIdentityObservationFailed
  | ProcessGroupObservationFailed

export type ProcessGroupSignalError =
  | ExactProcessIdentityObservationFailed
  | ProcessGroupSignalPermissionDenied
  | ProcessGroupSignalFailed

export type ProcessGroupStopError =
  | ExactProcessIdentityObservationFailed
  | ProcessGroupObservationFailed
  | ProcessGroupSignalPermissionDenied
  | ProcessGroupSignalFailed
  | ProcessGroupAbsenceUnproven
