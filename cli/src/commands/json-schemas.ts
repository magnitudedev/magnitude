import {
  CandidateInspectionSchema,
  CandidateVersionSchema,
  CatalogLocalModelSchema,
  CatalogLocalModelServingStateSchema,
  CustomLocalModelSchema,
  CustomLocalModelServingStateSchema,
  CustomModelAcquisitionStateSchema,
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
} from "@magnitudedev/sdk"
import { Schema, Struct } from "effect"

/*
 * `--json` documents carry the product facts the text form shows (design/cli/command-surface.md),
 * never the service's internal identities: assessment and environment IDs, memory-domain,
 * accelerator and device identities, managed paths and file ownership, or raw ranking inputs.
 * Catalog documents also carry no acquisition or residency state. These schemas mirror the
 * protocol's without those fields; encoding a protocol value through them drops the fields, so
 * callers pass protocol values unchanged.
 */

const optional = <A, I>(schema: Schema.Schema<A, I, never>) => Schema.optionalWith(schema, { as: "Option", exact: true })

const MemoryJsonSchema = Schema.Struct(Struct.pick(LocalModelMemorySchema.from.fields, "totalRequiredBytes", "requiredSystemMemoryBytes"))

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
const AllocationJsonSchema = Schema.Struct({
  ...ModelInstanceAllocationSchema.fields,
  memoryDomains: Schema.Array(Schema.Struct(Struct.omit(ModelInstanceAllocationSchema.fields.memoryDomains.value.fields, "memoryDomainId"))),
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
  Schema.Struct({ ...ready.fields, allocation: AllocationJsonSchema }),
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
  Schema.Struct({ ...removing.fields, ...InstalledJsonFields }),
  Schema.Struct({ ...removeFailed.fields, ...InstalledJsonFields }),
] as const
const AcquisitionStateJsonSchema = Schema.Union(...managedAcquisitionJsonMembers)

const external = CustomModelAcquisitionStateSchema.members[10]
const unavailable = CustomModelAcquisitionStateSchema.members[11]
const CustomAcquisitionStateJsonSchema = Schema.Union(
  ...managedAcquisitionJsonMembers,
  Schema.Struct({ ...external.fields, ...InstalledJsonFields }),
  Schema.Struct({ ...unavailable.fields, installation: InstallationJsonSchema }),
)

const [catalogAssessing, catalogFailed, catalogAssessedFits, catalogAssessedDoesNotFit] = CatalogLocalModelServingStateSchema.members
const CatalogServingStateJsonSchema = Schema.Union(
  catalogAssessing,
  catalogFailed,
  Schema.Struct({ ...Struct.omit(catalogAssessedFits.fields, "rankingScores"), assessment: FitsJsonSchema }),
  Schema.Struct({ ...catalogAssessedDoesNotFit.fields, assessment: DoesNotFitJsonSchema }),
)

const [customAssessing, customFailed, customAssessed] = CustomLocalModelServingStateSchema.members
const CustomServingStateJsonSchema = Schema.Union(
  customAssessing,
  customFailed,
  Schema.Struct({ ...customAssessed.fields, assessment: LocalModelAssessmentJsonSchema }),
)

const CatalogDataJsonSchema = Schema.Struct(Struct.omit(LocalModelCatalogDataSchema.fields, "fidelityRank"))

/** A catalog model as catalog commands show it: curated and machine-specific evidence, nothing on disk. */
export const CatalogModelJsonSchema = Schema.Struct({
  ...Struct.omit(CatalogLocalModelSchema.fields, "acquisitionState"),
  servingState: CatalogServingStateJsonSchema,
  catalogData: CatalogDataJsonSchema,
})

export const CatalogLocalModelJsonSchema = Schema.Struct({
  ...CatalogLocalModelSchema.fields,
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
  ...LocalInferenceHardwareSchema.from.fields,
  accelerators: Schema.Array(Schema.Struct(Struct.omit(LocalInferenceAcceleratorSchema.fields, "acceleratorId", "memoryDomainId"))),
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
