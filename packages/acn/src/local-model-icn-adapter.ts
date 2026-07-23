import { Effect, ParseResult, Schema } from "effect"
import type {
  DownloadAttempt,
  ModelOfferingTarget,
  ModelPackage,
  ModelPackageInspection,
  RecommendableModel,
  ServingProfile,
} from "@magnitudedev/protocol"
import {
  DownloadAttemptSchema,
  ModelOfferingTargetSchema,
  ModelPackageInspectionSchema,
  ModelPackageSchema,
  RecommendableModelSchema,
  ServingProfileSchema,
} from "@magnitudedev/protocol"
import type {
  DownloadAttempt as NativeDownloadAttempt,
  ModelOfferingTarget as NativeModelOfferingTarget,
  ModelPackageInspection as NativeModelPackageInspection,
  ModelPackage as NativeModelPackage,
  ModelPackageOperand,
  ModelTargetInput,
  RecommendableModel as NativeRecommendableModel,
  ServingProfile as NativeServingProfile,
} from "@magnitudedev/icn"
import {
  ModelOfferingTarget as NativeModelOfferingTargetSchema,
  ModelPackage as NativeModelPackageSchema,
  ModelTargetInput as NativeModelTargetInputSchema,
} from "@magnitudedev/icn"

export const modelPackageFromIcn = (
  modelPackage: NativeModelPackage,
): Effect.Effect<ModelPackage, ParseResult.ParseError> =>
  Schema.validate(ModelPackageSchema)(modelPackage)

export const modelPackageToIcn = (
  modelPackage: ModelPackage,
): Effect.Effect<NativeModelPackage, ParseResult.ParseError> =>
  Schema.validate(NativeModelPackageSchema)(modelPackage)

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
  parallelSequences: profile.parallelSequences,
})

export const offeringTargetFromIcn = (
  target: NativeModelOfferingTarget,
): Effect.Effect<ModelOfferingTarget, ParseResult.ParseError> =>
  Schema.validate(ModelOfferingTargetSchema)(target)

export const offeringTargetToIcn = (
  target: ModelOfferingTarget,
): Effect.Effect<NativeModelOfferingTarget, ParseResult.ParseError> =>
  Schema.validate(NativeModelOfferingTargetSchema)(target)

export const recommendableModelFromIcn = (
  model: NativeRecommendableModel,
): Effect.Effect<RecommendableModel, ParseResult.ParseError> =>
  Schema.validate(RecommendableModelSchema)(model)

export const downloadAttemptFromIcn = (
  attempt: NativeDownloadAttempt,
): Effect.Effect<DownloadAttempt, ParseResult.ParseError> =>
  Schema.validate(DownloadAttemptSchema)(attempt)

export const targetToIcn = (
  target: ModelOfferingTarget,
  installedPackageIds: ReadonlySet<string>,
): Effect.Effect<ModelTargetInput, ParseResult.ParseError> => {
  const operand = (modelPackage: ModelPackage): ModelPackageOperand =>
    installedPackageIds.has(modelPackage.id)
      ? { _tag: "Installed", packageId: modelPackage.id }
      : { _tag: "SourceBacked", package: modelPackage }
  const input = target._tag === "Package"
    ? { _tag: "Package" as const, package: operand(target.package) }
    : {
        _tag: "SpeculativeDecodingPair" as const,
        target: operand(target.target),
        draft: operand(target.draft),
      }
  return Schema.validate(NativeModelTargetInputSchema)(input)
}
