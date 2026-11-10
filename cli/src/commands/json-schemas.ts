import {
  CandidateInspectionSchema,
  CandidateVersionSchema,
  CatalogLocalModelFields,
  CustomLocalModelSchema,
  CustomLocalModelServingStateSchema,
  CustomModelAcquisitionStateSchema,
  HuggingFaceRecommendedVersionSchema,
  HuggingFaceSearchResultSchema,
  LocalInferenceAcceleratorSchema,
  LocalInferenceHardwareSchema,
  LocalInferenceMemoryDomainSchema,
  LocalModelAcquisitionStateSchema,
  LocalModelCatalogDataSchema,
  LocalModelDoesNotFitAssessmentSchema,
  LocalModelFitsAssessmentSchema,
  LocalModelMemorySchema,
  LocalModelUnsupportedAssessmentSchema,
  ModelInstallationSchema,
  ModelInstanceAllocationSchema,
  ModelLoadDeviceSchema,
  ModelLoadPlanSchema,
  ModelOptimizationProgressSchema,
  ModelResidencySchema,
  ModelStoppingAllocationSchema,
  ModelStreamingStatusSchema,
  StreamingHostStatusSchema,
  StreamingTierStatusSchema,
  CatalogLocalModelServingStateSchema,
} from "@magnitudedev/sdk"
import { Schema, Struct } from "effect"

/*
 * `--json` documents carry the product facts the text form shows,
 * never the service's internal identities: assessment and environment IDs, memory-domain,
 * accelerator and device identities, or managed paths and file ownership.
 * Catalog documents also carry no acquisition or residency state. These schemas mirror the
 * protocol's without those fields; encoding a protocol value through them drops the fields, so
 * callers pass protocol values unchanged.
 */

const optional = <A, I>(schema: Schema.Schema<A, I, never>) => Schema.optionalWith(schema, { as: "Option", exact: true })

const MemoryJsonSchema = Schema.Struct(Struct.pick(LocalModelMemorySchema.from.fields, "totalRequiredBytes", "requiredSystemMemoryBytes", "breakdown"))

const FitsJsonSchema = Schema.Struct({
  ...Struct.omit(LocalModelFitsAssessmentSchema.fields, "assessmentId", "environmentId"),
  memory: MemoryJsonSchema,
})
const DoesNotFitJsonSchema = Schema.Struct(Struct.omit(
  LocalModelDoesNotFitAssessmentSchema.from.fields,
  "assessmentId",
  "environmentId",
  "memoryDomains",
  "limitingResource",
))
const UnsupportedJsonSchema = Schema.Struct(Struct.omit(LocalModelUnsupportedAssessmentSchema.fields, "environmentId"))

export const LocalModelAssessmentJsonSchema = Schema.Union(FitsJsonSchema, DoesNotFitJsonSchema, UnsupportedJsonSchema)

const LoadDeviceJsonSchema = Schema.Struct(Struct.omit(ModelLoadDeviceSchema.fields, "deviceId"))
const LoadPlanJsonSchema = Schema.Struct({ ...ModelLoadPlanSchema.fields, device: LoadDeviceJsonSchema })
const StreamingJsonSchema = Schema.Struct({
  ...ModelStreamingStatusSchema.fields,
  tiers: Schema.Array(Schema.Struct(Struct.omit(StreamingTierStatusSchema.fields, "memoryDomainId"))),
  host: optional(Schema.Struct(Struct.omit(StreamingHostStatusSchema.fields, "memoryDomainId"))),
})
const AllocationJsonSchema = Schema.Struct({
  ...ModelInstanceAllocationSchema.fields,
  memoryDomains: Schema.Array(Schema.Struct(Struct.omit(ModelInstanceAllocationSchema.fields.memoryDomains.value.fields, "memoryDomainId"))),
  streaming: optional(StreamingJsonSchema),
})

const [stoppingPlanned, stoppingResident] = ModelStoppingAllocationSchema.members
const StoppingAllocationJsonSchema = Schema.Union(
  Schema.Struct({ ...stoppingPlanned.fields, allocation: optional(LoadPlanJsonSchema) }),
  Schema.Struct({ ...stoppingResident.fields, allocation: AllocationJsonSchema }),
)

const [unloaded, stopped, requested, loading, ready, stopping, residencyFailed] = ModelResidencySchema.members
const ResidencyJsonSchema = Schema.Union(
  unloaded,
  stopped,
  requested,
  Schema.Struct({ ...loading.fields, plannedAllocation: optional(LoadPlanJsonSchema) }),
  Schema.Struct({ ...ready.fields, allocation: AllocationJsonSchema, device: LoadDeviceJsonSchema }),
  Schema.Struct({ ...stopping.fields, allocation: StoppingAllocationJsonSchema }),
  residencyFailed,
)

const InstallationJsonSchema = Schema.Struct(Struct.pick(ModelInstallationSchema.members[1].fields, "installedBytes"))
const InstalledJsonFields = { installation: InstallationJsonSchema, residencyState: ResidencyJsonSchema } as const
const OptimizationProgressJsonSchema = Schema.Struct({
  ...ModelOptimizationProgressSchema.from.fields,
  device: optional(LoadDeviceJsonSchema),
})

const [
  notInstalled,
  installing,
  installFailed,
  installed,
  updateAvailable,
  updating,
  updateFailed,
  optimizing,
  optimizationPending,
  removing,
  removeFailed,
] = LocalModelAcquisitionStateSchema.members
const managedAcquisitionJsonMembers = [
  notInstalled,
  installing,
  installFailed,
  Schema.Struct({ ...installed.fields, ...InstalledJsonFields }),
  Schema.Struct({ ...updateAvailable.fields, ...InstalledJsonFields }),
  Schema.Struct({ ...updating.fields, ...InstalledJsonFields }),
  Schema.Struct({ ...updateFailed.fields, ...InstalledJsonFields }),
  Schema.Struct({ ...optimizing.fields, ...InstalledJsonFields, progress: OptimizationProgressJsonSchema }),
  Schema.Struct({ ...optimizationPending.fields, ...InstalledJsonFields }),
  Schema.Struct({ ...removing.fields, ...InstalledJsonFields }),
  Schema.Struct({ ...removeFailed.fields, ...InstalledJsonFields }),
] as const
const AcquisitionStateJsonSchema = Schema.Union(...managedAcquisitionJsonMembers)

const external = CustomModelAcquisitionStateSchema.members[11]
const unavailable = CustomModelAcquisitionStateSchema.members[12]
const CustomAcquisitionStateJsonSchema = Schema.Union(
  ...managedAcquisitionJsonMembers,
  Schema.Struct({ ...external.fields, ...InstalledJsonFields }),
  Schema.Struct({ ...unavailable.fields, installation: InstallationJsonSchema }),
)

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

const CatalogDataJsonSchema = LocalModelCatalogDataSchema

/** A catalog model as catalog commands show it: curated and machine-specific evidence, nothing on disk. */
export const CatalogModelJsonSchema = Schema.Struct({
  _tag: Schema.tag("Catalog"),
  ...Struct.omit(CatalogLocalModelFields, "acquisitionState"),
  servingState: CatalogServingStateJsonSchema,
  catalogData: CatalogDataJsonSchema,
})

export const CatalogLocalModelJsonSchema = Schema.Struct({
  _tag: Schema.tag("Catalog"),
  ...CatalogLocalModelFields,
  servingState: CatalogServingStateJsonSchema,
  catalogData: CatalogDataJsonSchema,
  acquisitionState: AcquisitionStateJsonSchema,
})

export const CustomLocalModelJsonSchema = Schema.Struct({
  ...CustomLocalModelSchema.from.fields,
  servingState: CustomServingStateJsonSchema,
  acquisitionState: CustomAcquisitionStateJsonSchema,
})

export const LocalModelJsonSchema = Schema.Union(CatalogLocalModelJsonSchema, CustomLocalModelJsonSchema)

export const LocalInferenceHardwareJsonSchema = Schema.Struct({
  ...Struct.omit(LocalInferenceHardwareSchema.from.fields, "activeDeviceId"),
  accelerators: Schema.Array(Schema.Struct({
    ...Struct.omit(LocalInferenceAcceleratorSchema.fields, "acceleratorId", "executors"),
    executors: Schema.Array(Schema.Struct(Struct.omit(LocalInferenceAcceleratorSchema.fields.executors.value.fields, "deviceId", "memoryDomainIds"))),
  })),
  memoryDomains: Schema.Array(Schema.Struct(Struct.omit(LocalInferenceMemoryDomainSchema.fields, "memoryDomainId"))),
})

const CandidateVersionJsonSchema = Schema.Struct({
  ...CandidateVersionSchema.fields,
  assessment: optional(LocalModelAssessmentJsonSchema),
})

export const CandidateInspectionJsonSchema = Schema.Struct({
  ...CandidateInspectionSchema.from.fields,
  versions: Schema.Array(CandidateVersionJsonSchema),
})

export const HuggingFaceSearchResponseJsonSchema = Schema.Struct({
  results: Schema.Array(Schema.Struct({
    ...HuggingFaceSearchResultSchema.fields,
    recommended: optional(Schema.Struct({
      ...HuggingFaceRecommendedVersionSchema.fields,
      assessment: LocalModelAssessmentJsonSchema,
    })),
  })),
  nextCursor: optional(Schema.String),
})
