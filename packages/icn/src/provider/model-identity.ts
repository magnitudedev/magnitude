import { createHash } from "node:crypto"
import { Schema } from "effect"
import { ProviderModelIdSchema, type ProviderModelId } from "@magnitudedev/ai"

export const NativeIcnModelIdSchema = Schema.String.pipe(
  Schema.minLength(1),
  Schema.maxLength(4096),
  Schema.brand("NativeIcnModelId"),
)
export type NativeIcnModelId = typeof NativeIcnModelIdSchema.Type

export const ModelRecipeConfigurationIdSchema = Schema.String.pipe(
  Schema.minLength(1),
  Schema.brand("ModelRecipeConfigurationId"),
)
export type ModelRecipeConfigurationId = typeof ModelRecipeConfigurationIdSchema.Type

export const ModelRecipeCatalogModelIdSchema = Schema.String.pipe(
  Schema.minLength(1),
  Schema.brand("ModelRecipeCatalogModelId"),
)

export const ModelArtifactFingerprintSchema = Schema.String.pipe(
  Schema.minLength(1),
  Schema.brand("ModelArtifactFingerprint"),
)

export const PrivateLocalModelIdSchema = Schema.String.pipe(
  Schema.minLength(1),
  Schema.brand("PrivateLocalModelId"),
)
export type PrivateLocalModelId = typeof PrivateLocalModelIdSchema.Type

const digest = (identity: string): string =>
  createHash("sha256").update(identity).digest("hex").slice(0, 32)

export const candidateLocalModelId = (
  recommendation: { readonly configurationId: ModelRecipeConfigurationId },
): PrivateLocalModelId => PrivateLocalModelIdSchema.make(`candidate_${digest(recommendation.configurationId)}`)

export const nativeLocalModelId = (nativeModelId: NativeIcnModelId): PrivateLocalModelId =>
  PrivateLocalModelIdSchema.make(`native_${digest(nativeModelId)}`)

export const localProviderModelId = (localModelId: PrivateLocalModelId): ProviderModelId =>
  ProviderModelIdSchema.make(`local:${localModelId}`)
