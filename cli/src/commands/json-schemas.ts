import {
  CandidateInspectionSchema,
  CandidateVersionSchema,
  CatalogLocalModelSchema,
  CatalogLocalModelServingStateSchema,
  CustomLocalModelSchema,
  CustomLocalModelServingStateSchema,
  LocalModelDoesNotFitAssessmentSchema,
  LocalModelFitsAssessmentSchema,
  LocalModelUnsupportedAssessmentSchema,
} from "@magnitudedev/sdk"
import { Schema, Struct } from "effect"

/*
 * `--json` documents carry the product facts the text form shows, never the service's internal
 * assessment identities. These schemas mirror the protocol's without those fields; encoding a
 * protocol value through them drops the fields, so callers pass protocol values unchanged.
 */

const FitsJsonSchema = Schema.Struct(Struct.omit(LocalModelFitsAssessmentSchema.fields, "assessmentId", "environmentId"))
const DoesNotFitJsonSchema = Schema.Struct(Struct.omit(LocalModelDoesNotFitAssessmentSchema.from.fields, "assessmentId", "environmentId"))
const UnsupportedJsonSchema = Schema.Struct(Struct.omit(LocalModelUnsupportedAssessmentSchema.fields, "environmentId"))

export const LocalModelAssessmentJsonSchema = Schema.Union(FitsJsonSchema, DoesNotFitJsonSchema, UnsupportedJsonSchema)

const [catalogAssessing, catalogFailed, catalogAssessedFits, catalogAssessedDoesNotFit] = CatalogLocalModelServingStateSchema.members
const CatalogServingStateJsonSchema = Schema.Union(
  catalogAssessing,
  catalogFailed,
  Schema.Struct({ ...catalogAssessedFits.fields, assessment: FitsJsonSchema }),
  Schema.Struct({ ...catalogAssessedDoesNotFit.fields, assessment: DoesNotFitJsonSchema }),
)

const [customAssessing, customFailed, customAssessed] = CustomLocalModelServingStateSchema.members
const CustomServingStateJsonSchema = Schema.Union(
  customAssessing,
  customFailed,
  Schema.Struct({ ...customAssessed.fields, assessment: LocalModelAssessmentJsonSchema }),
)

export const CatalogLocalModelJsonSchema = Schema.Struct({
  ...CatalogLocalModelSchema.fields,
  servingState: CatalogServingStateJsonSchema,
})

export const CustomLocalModelJsonSchema = Schema.Struct({
  ...CustomLocalModelSchema.from.fields,
  servingState: CustomServingStateJsonSchema,
})

export const LocalModelJsonSchema = Schema.Union(CatalogLocalModelJsonSchema, CustomLocalModelJsonSchema)

const CandidateVersionJsonSchema = Schema.Struct({
  ...CandidateVersionSchema.fields,
  assessment: Schema.optionalWith(LocalModelAssessmentJsonSchema, { as: "Option", exact: true }),
})

export const CandidateInspectionJsonSchema = Schema.Struct({
  ...CandidateInspectionSchema.from.fields,
  versions: Schema.Array(CandidateVersionJsonSchema),
})
