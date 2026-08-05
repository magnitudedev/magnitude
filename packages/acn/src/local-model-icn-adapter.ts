import { Effect, Option, ParseResult, Schema } from "effect"
import type {
  DownloadAttempt,
  ModelOfferingTarget,
  ModelPackage,
  ModelPackageInspection,
  ModelServingConfiguration,
  RecommendableModel,
  ServingProfile,
} from "@magnitudedev/acn-protocol"
import {
  DownloadAttemptSchema,
  ModelOfferingTargetSchema,
  ModelPackageInspectionSchema,
  ModelPackageSchema,
  RecommendableModelSchema,
  ServingProfileSchema,
} from "@magnitudedev/acn-protocol"
import type {
  DownloadAttempt as NativeDownloadAttempt,
  ModelOfferingTarget as NativeModelOfferingTarget,
  ModelPackageInspection as NativeModelPackageInspection,
  ModelPackage as NativeModelPackage,
  ModelServingConfiguration as NativeModelServingConfiguration,
  ModelTargetInput,
  RecommendableModel as NativeRecommendableModel,
  ServingProfile as NativeServingProfile,
} from "@magnitudedev/icn-protocol/schemas"

const normalizeModelPackageFromIcn = (
  modelPackage: NativeModelPackage,
) => ({
  ...modelPackage,
  files: modelPackage.files.map((file) => ({
    ...file,
    tensorStorageBytes: Option.flatMap(file.tensorStorageBytes, Option.fromNullable),
  })),
})

const normalizeOfferingTargetFromIcn = (
  target: NativeModelOfferingTarget,
) => target._tag === "Package"
  ? { ...target, package: normalizeModelPackageFromIcn(target.package) }
  : {
      ...target,
      target: normalizeModelPackageFromIcn(target.target),
      draft: normalizeModelPackageFromIcn(target.draft),
    }
import {
  ModelOfferingTarget as NativeModelOfferingTargetSchema,
  ModelPackage as NativeModelPackageSchema,
  ModelTargetInput as NativeModelTargetInputSchema,
} from "@magnitudedev/icn-protocol/schemas"

export const modelPackageFromIcn = (
  modelPackage: NativeModelPackage,
): Effect.Effect<ModelPackage, ParseResult.ParseError> =>
  Schema.validate(ModelPackageSchema)(normalizeModelPackageFromIcn(modelPackage))

export const modelPackageToIcn = (
  modelPackage: ModelPackage,
): Effect.Effect<NativeModelPackage, ParseResult.ParseError> =>
  Schema.encode(ModelPackageSchema)(modelPackage).pipe(
    Effect.flatMap(Schema.decodeUnknown(NativeModelPackageSchema)),
  )

export const packageInspectionFromIcn = (
  inspection: NativeModelPackageInspection,
): Effect.Effect<ModelPackageInspection, ParseResult.ParseError> =>
  Schema.validate(ModelPackageInspectionSchema)(inspection)

export const servingProfileFromIcn = (
  profile: NativeServingProfile,
): Effect.Effect<ServingProfile, ParseResult.ParseError> =>
  Schema.validate(ServingProfileSchema)(profile)

export const servingProfileToIcn = (profile: ServingProfile): NativeServingProfile => ({
  contextLength: profile.contextLength,
})

export const offeringTargetFromIcn = (
  target: NativeModelOfferingTarget,
): Effect.Effect<ModelOfferingTarget, ParseResult.ParseError> =>
  Schema.validate(ModelOfferingTargetSchema)(normalizeOfferingTargetFromIcn(target))

export const offeringTargetToIcn = (
  target: ModelOfferingTarget,
): Effect.Effect<NativeModelOfferingTarget, ParseResult.ParseError> =>
  Schema.encode(ModelOfferingTargetSchema)(target).pipe(
    Effect.flatMap(Schema.decodeUnknown(NativeModelOfferingTargetSchema)),
  )

export const modelServingConfigurationToIcn = (
  configuration: ModelServingConfiguration,
): Effect.Effect<NativeModelServingConfiguration, ParseResult.ParseError> =>
  offeringTargetToIcn(configuration.target).pipe(
    Effect.map((target) => ({
      id: configuration.id,
      target,
      profile: servingProfileToIcn(configuration.profile),
    })),
  )

export const recommendableModelFromIcn = (
  model: NativeRecommendableModel,
): Effect.Effect<RecommendableModel, ParseResult.ParseError> =>
  Schema.validate(RecommendableModelSchema)({
    ...model,
    target: normalizeOfferingTargetFromIcn(model.target),
  })

export const downloadAttemptFromIcn = (
  attempt: NativeDownloadAttempt,
): Effect.Effect<DownloadAttempt, ParseResult.ParseError> =>
  Schema.validate(DownloadAttemptSchema)(attempt._tag === "Downloading"
    ? {
        ...attempt,
        bytesPerSecond: Option.flatMap(attempt.bytesPerSecond, Option.fromNullable),
      }
    : attempt)

export const targetToIcn = (
  target: ModelOfferingTarget,
  installedPackageIds: ReadonlySet<string>,
): Effect.Effect<ModelTargetInput, ParseResult.ParseError> => {
  const operand = (modelPackage: ModelPackage) =>
    installedPackageIds.has(modelPackage.id)
      ? Effect.succeed({ _tag: "Installed" as const, packageId: modelPackage.id })
      : Schema.encode(ModelPackageSchema)(modelPackage).pipe(
          Effect.map((encoded) => ({ _tag: "SourceBacked" as const, package: encoded })),
        )
  return Effect.gen(function* () {
    const input = target._tag === "Package"
      ? { _tag: "Package" as const, package: yield* operand(target.package) }
      : {
          _tag: "SpeculativeDecodingPair" as const,
          target: yield* operand(target.target),
          draft: yield* operand(target.draft),
        }
    return yield* Schema.decodeUnknown(NativeModelTargetInputSchema)(input)
  })
}
