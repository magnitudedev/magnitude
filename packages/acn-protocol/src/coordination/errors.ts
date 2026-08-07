import { Data } from "effect"

export class AcnProcessStoreUnavailable extends Data.TaggedError("AcnProcessStoreUnavailable")<{
  readonly operation: string
  readonly path: string
  readonly message: string
}> {}

export class AcnProcessStoreInvalid extends Data.TaggedError("AcnProcessStoreInvalid")<{
  readonly path: string
  readonly message: string
}> {}

export class AcnProcessStoreBusy extends Data.TaggedError("AcnProcessStoreBusy")<{
  readonly operation: string
  readonly path: string
}> {}

export type AcnProcessStoreError =
  | AcnProcessStoreUnavailable
  | AcnProcessStoreInvalid
  | AcnProcessStoreBusy

export class ExactProcessInspectionFailed extends Data.TaggedError("ExactProcessInspectionFailed")<{
  readonly pid: number
  readonly operation: string
  readonly message: string
}> {}
